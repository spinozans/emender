# X39 part 9 — four trained arms; compression receipts

**Arm 4′ completed first in part 9:** 3052 B64 updates, 100,007,936 target
tokens, λ=.01, seed 3401. Its fixed-window perplexity is
**6.93538665**, ratio **1.01151754** to B64 control
6.8564175640. Per-head log-log peak-occupancy slopes at
T=512/1024/2048/4096 are **0.0945/0.0550/0.1275/0.1535**.
The registered verdict is **strong occupancy compression screen passes with registered quality**. These finite-window occupancy
measurements do not independently establish semantic information content.

| Arm | Registered PPL ratio / slope | Measured PPL / ratio | Per-head slope | Disposition |
|---|---|---|---|---|
| 1 decay horizon128 | ≤1.05 / ≤.5 | 6.92328267 / 1.00975219 | 0.0000/0.0000/0.0000/0.0000 | quality passes; strong slope screen passes |
| 2 gate cost .01 | ≤1.02 / ≤.5 | 6.91114998 / 1.00798265 | 0.9575/0.9602/0.9294/0.9402 | quality passes; strong slope screen fails |
| 3 poverty S128 | ≤1.02 / ≤.5 | 6.89398492 / 1.00547915 | 0.0000/0.0000/0.0000/0.0000 | capacity-capped; rich transfer fails |
| 3 same weights S2944 | ≤1.02 / ≤.5 | 6.87446791 / 1.00263262 | 1.0000/1.0000/1.0000/1.0000 | quality passes; strong slope screen fails |
| 4′ shared semantic + match + cost .01 | ≤1.02 / ≤.5 | 6.93538665 / 1.01151754 | 0.0945/0.0550/0.1275/0.1535 | quality passes; strong slope screen passes |


All training endpoints above are 100,007,936 tokens. Quality and slope bands
are pre-registered acceptance screens, not predictions. Poverty poor/rich are
one trained arm at two deployment budgets, with identical checkpoint hash.
Part-7/8 provenance qualifications stand exactly as receipted: recorded hashes
identify disk snapshots and cannot conclusively identify cached bytecode.
This final synthesis adds no retrospective certification or reconciliation.

## Compression conclusions

1. **Decay is the built-in occupancy regulator.** A 128-token untouched-slot
   fade horizon and one birth per two tokens yield the observed equilibrium
   of 64 slots/head. Occupancy slope is approximately zero by construction,
   while allocation remains .5/token/head. This is expiration of old state,
   not evidence of learned semantic deduplication. `part7_decay_shape.json`
   preserves 128-token NLL bins and context ratios; full position curves and
   occupancy profiles are in `part7_decay_evaluation.json`.
2. **Gate cost changes the constant without demonstrating a different
   asymptotic exponent.** Allocations fall to about .14–.30/token/head at
   T4096, but measured slopes .9294–.9602 fail the ≤.5 strong-compression
   screen. The finite-range estimates are near one; they do not prove an
   exact asymptotic exponent. The .01 operating point passes its 2% quality
   band; .1 failed the matched 20M selection pilot.
3. **Poverty thrift is situational, not dispositional in this experiment.**
   The same weights give slopes approximately 0 at S128 and 1 at S2944.
   At S128 pressure rises .5→.9375 over these lengths; the flat curve is a
   capacity cap. Rich-budget allocations return to .5/token/head. Strong
   compression does not transfer to rich deployment.
4. **Arm 4′: strong occupancy compression screen passes with registered quality.** Shared W_in, detached recurrent match evidence and
   gate cost are now trained rather than inferred from the hand-set witness.
   Peak occupancies over the registered lengths are 9–14 slots/head, far below
   S2944; pressure is zero. At T4096 allocations are .00256–.00305/token/head.
   This supports the registered conceptual-cache occupancy hypothesis over
   these lengths, with the retained-information limits below.
   The table reports its measured outcome without substituting retrieval
   success or a capacity cap for conceptual-cache evidence.

## Arm 4′ execution and evidence

`part9_semantic_parity.json` records independent sequential X31 state/evidence
and exact gate-decision parity at B64/H4/K64/V128/S2944/T512. Full-scan
forward/state/trace/pressure and gradients match X31 bitwise, and ReplayBackward
is asserted. The evidence score comparison allows rtol 2e-5 / atol 2e-6 for
PyTorch versus tiled reduction. This is a targeted production-shape test,
not exhaustive adversarial coverage. An initial harness attempt used an
incorrect trace field name and failed before training; its console is retained.
The corrected gate passed before the accepted run. A post-training replay
recheck explicitly compares every trace field, including the informational
matched index and POP-fired bit; it removes the harness's conditional field
skipping. Exact pre-training harness bytes and its passing certificate are
preserved as `runs/semantic_full/pretraining_parity.py` and
`part9_semantic_parity_pretraining.json`. Model/training code is unchanged.
Existing CPU suite: 11 passed.

`part9_gate_witness.json` reruns both required alias ablations on the actual
optimized wire at corrected READ scale: semantic+wired gives novel/repeat
allocation 1/0; identity+wired and semantic+unwired give 1/1. This remains a
hand-set circuit witness, not a trained-language ablation. The learned-parameter
receipt confirms a single shared 64×256 W_in moved from initialization
(distance 3.407416); its off-identity
support norm 2.777794 shows learning
beyond simple decay. Match coefficients finish at approximately −1.785.

Every accepted update asserts ReplayBackward and finite loss/gradient norm.
Raw CE/loss/gate-strength/norm records are `runs/semantic_full/curve.jsonl`;
checkpoint validation receipts are `part9_semantic_full_<step>.json` and the
standalone curve is `part9_semantic_curve.svg`. Training, checkpoint hash,
source/config/data fingerprints and exact pre-training protocol bytes are
retained. Physical broker card 1 was used; no forbidden
or certified-held card was requested. Checkpoint weights remain local and
ignored; no .pt is committed.

Parameter receipt: semantic 30,220,564 total /
21,797,648 trainable; widened base
30,220,615 total, relative total gap
1.6875897e-06. The widened base is untrained;
the registered B64 quality comparator remains the previously trained control.
Architecture and initialization are specified before training in
`runs/semantic_full/registered_protocol.md`. Existing append error placement
is preserved. The cost applies to the actual wired sigmoid.

`part9_semantic_full_evaluation.json` retains two windows at every length,
full position NLL, per-head occupancy profiles, successful allocations,
requests, write-enable and sigmoid strengths, pressure/R-CAP/W64, first 16
story profiles with truncation flags, and seven depth probes. These are the
same probes as the control. `part9_adjudication.json` derives the final table.

| Arm | T | Allocations/token/head | Pressure fraction/head | Mean occupancy/head | Context PPL/control |
|---|---|---|---|---|---|
| decay | 512 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 55.7480/55.7480/55.7480/55.7480 | 1.009874 |
| decay | 1024 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 59.6240/59.6240/59.6240/59.6240 | 0.961139 |
| decay | 2048 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 61.5620/61.5620/61.5620/61.5620 | 0.910427 |
| decay | 4096 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 62.5310/62.5310/62.5310/62.5310 | 0.928291 |
| gate_0.01_full | 512 | 0.3105/0.2236/0.1689/0.2314 | 0.0000/0.0000/0.0000/0.0000 | 79.0654/58.5283/41.9639/61.2793 | 1.033041 |
| gate_0.01_full | 1024 | 0.3047/0.2134/0.1621/0.2280 | 0.0000/0.0000/0.0000/0.0000 | 150.7798/109.2959/84.9170/120.8555 | 1.008674 |
| gate_0.01_full | 2048 | 0.3022/0.2131/0.1577/0.2229 | 0.0000/0.0000/0.0000/0.0000 | 314.0076/224.6636/170.6831/231.6401 | 0.975756 |
| gate_0.01_full | 4096 | 0.2979/0.2124/0.1404/0.2119 | 0.0000/0.0000/0.0000/0.0000 | 607.4150/439.6676/297.8640/457.3630 | 0.990245 |
| poverty | 512 | 0.2500/0.2500/0.2500/0.2500 | 0.5000/0.5000/0.5000/0.5000 | 96.2500/96.2500/96.2500/96.2500 | 1.013493 |
| poverty | 1024 | 0.1250/0.1250/0.1250/0.1250 | 0.7500/0.7500/0.7500/0.7500 | 112.1250/112.1250/112.1250/112.1250 | 1.004890 |
| poverty | 2048 | 0.0625/0.0625/0.0625/0.0625 | 0.8750/0.8750/0.8750/0.8750 | 120.0625/120.0625/120.0625/120.0625 | 0.991460 |
| poverty | 4096 | 0.0312/0.0312/0.0312/0.0312 | 0.9375/0.9375/0.9375/0.9375 | 124.0312/124.0312/124.0312/124.0312 | 0.994263 |
| poverty_rich | 512 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 128.5000/128.5000/128.5000/128.5000 | 1.010187 |
| poverty_rich | 1024 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 256.5000/256.5000/256.5000/256.5000 | 1.001422 |
| poverty_rich | 2048 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 512.5000/512.5000/512.4048/512.5000 | 0.980498 |
| poverty_rich | 4096 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 1024.5000/1024.5000/1024.4524/1024.5000 | 0.983249 |
| semantic_full | 512 | 0.0186/0.0176/0.0186/0.0166 | 0.0000/0.0000/0.0000/0.0000 | 8.2412/6.2324/7.5703/7.0615 | 1.014694 |
| semantic_full | 1024 | 0.0107/0.0088/0.0093/0.0088 | 0.0000/0.0000/0.0000/0.0000 | 9.3955/7.5703/8.3311/7.3989 | 1.003001 |
| semantic_full | 2048 | 0.0061/0.0051/0.0059/0.0049 | 0.0000/0.0000/0.0000/0.0000 | 10.5222/8.9434/10.1726/8.3120 | 0.962082 |
| semantic_full | 4096 | 0.0031/0.0026/0.0027/0.0029 | 0.0000/0.0000/0.0000/0.0000 | 10.8898/9.3568/9.3065/10.3918 | 0.991428 |


Arm 4′'s 16-story median peak occupancy per head is **7.0000/6.0000/6.0000/6.5000**;
maximum is **13.0000/10.0000/9.0000/11.0000**. The complete per-token profiles, story
lengths and truncation flags are retained in its evaluation receipt.
Depth bins use very few paired prompts; these measurements do not by
themselves establish general recall or semantic information preservation.

| Depth bin | Control accuracy | Arm 4′ accuracy | Arm 4′ valid−corrupted logprob |
|---|---|---|---|
| ID | 0.5000 | 0.5000 | 0.005429 |
| 2x | 0.5000 | 0.5000 | 0.007647 |
| 4x | 0.5000 | 0.5000 | 0.010528 |
| 9x | 0.5000 | 0.5000 | -0.017224 |


## Structural-lessons ledger, parts 1–9

| Part | Receipted lesson | Limit |
|---|---|---|
| 1 | Append amplitude scales stored values after routing; it is not allocation probability. | A differentiable gate does not imply differentiable allocation count. |
| 2 | Penalizing continuous write-enable sigmoid can cross the hard threshold. | Synthetic controller evidence does not choose a language-model lambda. |
| 3 | Scarcity supplies loss through capacity-limited recall; semantic retrieval alone leaves writes enabled. | Retrieval and write suppression are separate circuit properties. |
| 4 | Hand-set semantic keys plus read-match wire suppress alias-repeat writes; both ablations fail. | Constructive witness is not trained natural-text evidence; its historical score scale differs. |
| 5 | Missing directive/control and optimized wire were explicit execution blockers in archived work. | An unrun arm is neither a null nor a compression result. |
| 6 | Exact-exposure B64 control resolves pilot/full quality denominators. | B8 control was not optimizer-update matched. |
| 7 | Decay equilibrium, gate-cost coefficient reduction, poor/rich transfer failure measured. | One seed; original source qualification retained. |
| 8 | Disk snapshot hashes do not identify already cached bytecode; current oracle parity passes. | This distinction is the final provenance disposition. |
| 9 | Optimized recurrent match wire certified, trained and evaluated with shared W_in. | strong occupancy compression screen passes with registered quality; finite windows do not prove general conceptual memory. |

Residual risks: One training seed; only two packed windows per length, 16 stories and seven depth probes. Fixed-window perplexity band does not establish general quality or retention beyond 4096. Semantic architecture changes frozen features and total parameters; widened base is a parameter receipt, not a newly trained control. No trained wire-only or semantic-only ablation isolates the causal contribution of learned W_in. Hard detached routing has no allocation-count gradient; match-evidence is detached and gate cost optimizes the wired sigmoid. Historical disk snapshots cannot establish cached loaded-code identity; existing qualification remains final. Rich pressure zero is
not an improvement over control, which also has zero pressure here. A
low slope caused by information loss is not by itself successful compression.

---

# X39 part 8 — provenance audit incomplete; semantic arm remains untrained

The requested source re-attestation cannot yet be made from the run receipts.
`receipts/part8_provenance.json` preserves every X36 **registered disk** hash
from all six completed run receipts, together with current-disk and HEAD
comparisons. These are not loaded-code hashes. Part 7 used one Python process;
`usage.py` records `fingerprint()` before invoking parity, and `cell_fused.py`
imports `backward.py` lazily. Later registrations can therefore hash new disk
bytes while training continues to execute an earlier cached module. The prior
integrity failure explicitly records that distinction. Re-labeling a later
registration as the actual training source would erase the unresolved issue.
No historical receipt was overwritten, no completed arm was retrained, and
no source-attestation gate was changed to passing.

The differing registered backward hashes are:

| Run | Registered backward.py SHA256 |
|---|---|
| decay; gate .001 pilot; gate .01 pilot | `2a658aeb26403dc6f24d7572d6f19399f509e7a94850b8f725e7a19869a0e1d9` |
| gate .1 pilot; gate .01 full; poverty | `0eb8f7ee6ba0e245516ec8e7f8fbe7069f1c3fe390686cafbbdd42e95a51b1cb` |

All six register replay_setup.py as
`8ee8c62130e838f5c7c0f532bb610a411de2bc0eab94130d200e13a77893d83f`.
Full per-file hashes and receipt digests are in the new JSON audit. Recovering
historical source bytes and establishing import-time identity remain needed
before a claim of equivalence at the actual historical implementations.
Matching current oracle outputs alone cannot establish that identity.

A fresh broker-leased production-shape B64/H4/K64/V128/S2944/T512 oracle test
**passed**, including forward/state/trace and gradient comparisons (1 test,
16.38 seconds, physical card 1). Evidence is
`receipts/part8_current_parity.{txt,xml}`. This validates the current working
sources at this shape; it does not certify both historical revisions or the
unimplemented semantic circuit. Inspection also found a working-tree change
in `forward.py` journal initial-bound selection, beyond the stated
backward/replay_setup changes. Upstream sources were left untouched.

The poverty receipts were independently cross-checked: S128 and S2944 use
checkpoint SHA256
`a3443c6ae9d348511e1bb22a5233531c1e190032782e5d14789de12ca0d11bbc`.
Their recorded slopes are respectively approximately 0 and 1 in every head.
This confirms consistency of the existing situational-thrift result, not a
fresh GPU deployment evaluation. All part-7 trained metrics and source
qualification below remain applicable.

**Arm 4' remains untrained.** No optimized recurrent match wire, semantic
oracle comparison, ReplayBackward training receipt, or natural-text verdict
was produced in this attempt. The four-arm acceptance contract is unmet.
Discovery found no `contact_supervisor` or Intercom tool, so the required
coordination channel could not be used. No alternative service was started.
No success-titled commit or push was made; that title would assert training
that has not happened. No checkpoint files were added.

Residual risks: unresolved loaded-source identity; no historical-revision
parity comparison; no semantic-arm result; the existing single-seed and small
probe-sample limits. An honest scientific null for arm 4' cannot be inferred
from an unimplemented/untrained arm.

---

# X39 part 7 — provisional three-arm measurements; source gate failed

## Part 7 measured result — three trainings complete; provenance failed; fourth unrun

**All new scientific measurements are provisional.** Final integrity checking
detected external edits to X36 `backward.py` and `replay_setup.py` during the
single Python training process. Initial decay registration matched the B64
control's hashes for these modules, but later registrations saw changed disk
bytes. Cached Python imports suggest that much loaded code stayed unchanged;
this is not a complete loaded-source attestation. The unchanged-source gate
**failed**. `receipts/part7_integrity.json` retains each registered/current hash
mismatch. No training is claimed fully certified from this evidence. Upstream
files were read-only for this task and were not restored or modified.

Arms 1–3 completed sequentially at B64, seed 3401, context 512, each with
100,007,936 target tokens. The three lambda pilots each completed 20,021,248
tokens. Corrected runs held broker card 0 exclusively; certified cards 2/3/4
and forbidden card 7 were excluded by the unchanged X39 wrapper.
**The four-arm task is incomplete:** arm 4' was not trained. This is an
implementation blocker, not a scientific null from a trained semantic model.

| Arm | Pre-registered | Measured PPL / control ratio | Per-head slope | Disposition |
|---|---|---|---|---|
| 1 decay, horizon 128 | 100M; PPL ratio <= 1.05; slope <= .5 | 6.92328267 / 1.00975219 | 0.0000/0.0000/0.0000/0.0000 | provisional compression screen passes; source attestation fails |
| 2 gate cost, lambda 0.01 | 100M; PPL ratio <= 1.02; slope <= .5 | 6.91114998 / 1.00798265 | 0.9575/0.9602/0.9294/0.9402 | quality passes; strong compression fails |
| 3 poverty, S128 | 100M; PPL ratio <= 1.02; slope <= .5 | 6.89398492 / 1.00547915 | 0.0000/0.0000/0.0000/0.0000 | capacity-capped; rich transfer is the disposition test |
| 3 same weights, S2944 | 100M; PPL ratio <= 1.02; slope <= .5 | 6.87446791 / 1.00263262 | 1.0000/1.0000/1.0000/1.0000 | quality passes; strong compression fails |
| 4' semantic keys + match wire | Operating lambda; 100M; parity + ReplayBackward | Not run | Unmeasured | Optimized recurrent wire absent; incomplete, not a trained null |


The control is 6.8564175640 at 100M and 14.6701488443 at the exact pilot
exposure. The selected lambda is **0.01**, the largest swept value passing
the registered threshold 14.96355182. The full run starts
independently; it is not a continuation of its pilot.

| Lambda | PPL at 20,021,248 tokens | Ratio to control | Within 2% |
|---|---|---|---|
| 0.001 | 14.67135383 | 1.00008214 | True |
| 0.01 | 14.78951122 | 1.00813641 | True |
| 0.1 | 15.21379435 | 1.03705794 | False |


`receipts/part7_lambda_interpolation.json` records descriptive log-lambda/PPL
boundary interpolation between bracketing pilots, with no extrapolation and
no invented unswept operating point. Validation checkpoint receipts and every
training loss/gate mean are retained as curves. The repeated .01 full-run
20M validation is 14.7947889612, versus the pilot's 14.7895112197; both pass.
Same-seed CUDA training is not claimed to be bit-identical across executions.

### Synthesis and structural-lessons ledger

The provisional decay measurements show non-capacity-limited sublinear retained memory at an acceptable
short-context quality cost. Its unchanged .5 allocation rate distinguishes
expiration from selective writing. Gate cost reduces writes substantially,
but does not produce the registered strong sublinear scaling. No strong compression transfers to rich deployment.
The poor/rich evaluations share the exact checkpoint hash, changing only the
deployment budget. Pressure zero in a rich budget is not itself an improvement:
the control also has zero pressure over these lengths.

| Structural lesson | Trained evidence and limit |
|---|---|
| Decay bounds retained memory without reducing writes | Every registered length peaks at 64 slots/head; slope about zero, allocation .5/token/head, zero rich pressure; short-context PPL costs .975%. This is recent retention, not semantic deduplication. |
| Write-enable cost reduces the allocation coefficient | Selected lambda .01 gives slopes .9294–.9602: fewer writes, but the <=.5 strong-compression criterion fails in all heads. PPL costs .798%. |
| Capacity caps can mimic compression | S128 occupancy must be interpreted alongside pressure and the same trained weights at S2944; low capped slope alone is not learned compression. |
| Semantic retrieval is not trained deduplication evidence | Prior constructive CPU wire passes its two ablations, but no optimized recurrent wire or trained semantic result exists. |


| Arm | T | Allocation/token/head (two-window mean) | Pressure fraction/head | Context PPL / control |
|---|---|---|---|---|
| decay | 512 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 1.009874 |
| decay | 1024 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 0.961139 |
| decay | 2048 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 0.910427 |
| decay | 4096 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 0.928291 |
| gate_0.01_full | 512 | 0.3105/0.2236/0.1689/0.2314 | 0.0000/0.0000/0.0000/0.0000 | 1.033041 |
| gate_0.01_full | 1024 | 0.3047/0.2134/0.1621/0.2280 | 0.0000/0.0000/0.0000/0.0000 | 1.008674 |
| gate_0.01_full | 2048 | 0.3022/0.2131/0.1577/0.2229 | 0.0000/0.0000/0.0000/0.0000 | 0.975756 |
| gate_0.01_full | 4096 | 0.2979/0.2124/0.1404/0.2119 | 0.0000/0.0000/0.0000/0.0000 | 0.990245 |
| poverty | 512 | 0.2500/0.2500/0.2500/0.2500 | 0.5000/0.5000/0.5000/0.5000 | 1.013493 |
| poverty | 1024 | 0.1250/0.1250/0.1250/0.1250 | 0.7500/0.7500/0.7500/0.7500 | 1.004890 |
| poverty | 2048 | 0.0625/0.0625/0.0625/0.0625 | 0.8750/0.8750/0.8750/0.8750 | 0.991460 |
| poverty | 4096 | 0.0312/0.0312/0.0312/0.0312 | 0.9375/0.9375/0.9375/0.9375 | 0.994263 |
| poverty_rich | 512 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 1.010187 |
| poverty_rich | 1024 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 1.001422 |
| poverty_rich | 2048 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 0.980498 |
| poverty_rich | 4096 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 0.983249 |


Full occupancy profiles (mean, peak, final), allocation/request/enable/sigmoid
rates, per-head pressure/R-CAP/W64 statistics, first 16 story profiles with
truncation flags, seven depth probes and all per-position context NLL are in
the per-arm `part7_*_evaluation.json` receipts. `part7_decay_shape.json` gives
128-token NLL bins and matched context ratios; raw position curves remain in
the evaluation receipts. The decay context ratios are about 1.0099/.9611/
.9104/.9283 at T=512/1024/2048/4096: longer-context degradation relative to
control is not observed on these registered windows.

### Validation, provenance and remaining work

Every invocation passed a fresh exact-configuration B64/H4/K64/V128/T512
comparison against certified X31: forward values, final state, routing traces
and gradients bitwise. This is the targeted production-shape gate, not a new
claim that the entire adversarial suite was rerun. Parameter receipts match
the B64 control's 21,913,872 total / 21,847,052 trainable tape parameters
against base_lm's 21,914,119 total. Every accepted update asserts ReplayBackward
and finite loss/gradient norm. All 11 existing X39 CPU tests pass.
`part7_integrity.json` verifies uninterrupted step/exposure sequences but
**fails unchanged-source provenance**. Each run retains the exact
`registered_protocol.md` bytes used for its fingerprint.

An initial implementation attempt was stopped after 460 updates / 15,073,280
tokens because its RNG-consuming model-construction order differed from the
control. It is archived under `receipts/part7_aborted_initialization`, excluded
from all results, and replaced by fresh runs matching control construction
order. Every gate run's first-step CE exactly equals the control's first CE.
No checkpoint files are committed. Pre-existing upstream working-tree edits
remain untouched; their source hashes identify the implementations used.

Arm 4' still requires an optimized recurrent pre-read-match-to-write-enable
lane and its exact oracle/replay certification. `part4_match.py` is explicitly
a sequential CPU witness; its 1/sqrt(K) evidence scale differs from certified
READ's 2/sqrt(K). The archived report also leaves error-term placement
unresolved. No architecture was silently substituted. Discovery by tool name
and description exposed no contact_supervisor/Intercom capability; no alternate
orchestration service was started. `part7_semantic_blocker.json` records this
remaining work. A title claiming four trained arms would misstate the result.

Residual risks: failed source provenance from concurrent external changes,
one training seed, small registered validation/probe samples,
same-seed numerical variation, and no learned semantic-arm evidence. Capacity
limits can destroy information while flattening slopes. Reported bands are
fixed-window comparisons and do not establish general language quality or
longer-than-4096 retention.

## Historical reports (superseded where arms 1–3 are described as unrun)

# X39 result — control trained; usage arms incomplete

## Part 6 — B64 control completed; four usage arms incomplete

The newly authorized B64 exact-exposure control **completed** 3052 updates,
100,007,936 target tokens, on broker card 1 in **831.07 seconds**. Its final
fixed-window validation perplexity is **6.8564175640**. At update 611 /
20,021,248 tokens its perplexity is **14.6701488443**; this is the denominator
for the registered 2% lambda decision. No X39 lambda pilot has been run, so
there is neither a selected lambda nor evidence for a no-qualifier null.

The amendment and its operator provenance are in RECEIPTS.md. Configuration
and data/source hashes are in `runs/control_b64/registration.json` and
`receipts/part6_control_training.json`. X36's optimized model preserves the
X34 skeleton, initialized weights, policies, seed, data order and optimizer;
only the registered batch change changes update grouping. Parameter totals
are 21,913,872 (21,847,052 trainable), versus base_lm's 21,914,119. Every one
of the 3052 loss graphs asserted ReplayBackward, and every gradient norm and
loss passed finite checks. The raw training curve is `runs/control_b64/curve.jsonl`;
validation checkpoints and `receipts/part6_control_curve.svg` retain its curve.
Checkpoint weights remain local and ignored by git.
`runs/control_b64/registered_protocol.md` preserves the exact pre-training
registration bytes and verified hash before the final dispositions were appended.

Before training, the fresh production-shape B64/H4/K64/V128/S2944/T512 X36
versus X31 forward/backward test passed in 16.25 seconds, leased on card 6.
Its XML and console are `receipts/part6_control_parity.{xml,txt}`. This test
compares the optimized implementation with the certified X31 implementation;
it is not an assertion that the entire adversarial suite was rerun. The
existing CPU X39 suite also passed all 11 tests.

`receipts/part6_control_evaluation.json` contains two windows at each registered
length, full per-position NLL, occupancy profiles, successful allocations,
requests, write-enable fractions, sigmoid strength, per-head pressure and
R-CAP/W64 statistics, the first 16 validation stories, and seven depth probes.
Evaluation verified that all training-recorded source hashes still matched.
The measured per-head occupancy slopes are **1, 1, 1, .9997886161**. Rich-budget
pressure is zero throughout the four packed-context evaluations. Allocation
is .5/token/head in most windows, with small reductions in head 4 at longer
contexts. Control context perplexities at T=512/1024/2048/4096 are
7.164688392/8.817347908/12.383386623/18.992085553; these use different windows
from the fixed short-context quality validation above.

### Completion limit and remaining ordered work

**The four usage trainings, X39 lambda sweep and all trained-arm adjudications
remain unrun.** The requested part-5 directive and blocker receipt are not in
the checkout or recent X39 history. Tool discovery by name and description
found no contact_supervisor/Intercom capability, so the required decision
channel cannot supply those inputs. No alternative service was started.
The part-6 instruction supplies enough detail to complete the control; it
does not supply the missing directive to execute the subsequent arms exactly.
In particular, the part-4 report leaves the optimized match wire and placement
of the error term unspecified. No architecture was silently substituted.

An additional source-level discrepancy should be resolved with that directive:
`part4_match.py` computes match evidence with scale 1/sqrt(K), while actual
certified READ uses the sharpened 2/sqrt(K). Its comment equating these scales
is inaccurate. The constructive repeat-suppression witness remains evidence
for its recorded circuit; it does not certify the trained match wire. The
historical receipt and its hashes are retained unchanged.

X37/X38 remain read-only. `receipts/part6_family_comparability.json` rechecks
recorded batch, seed, context, tokenizer, packed train/validation data hashes
and exact evaluation exposure. X37 b/c at 100M have perplexity ratios
.9844033740/.9761630030 to the new B64 control, both within the no-more-than-2%
worse band. X37 d lacks a completed endpoint. X38 pilots at 20M have ratios
.9973254272/.9989068527/1.0102315072/1.0222863909 for costs
0/.001/.01/.1: the first three pass 2%, .1 fails. Its .01 full run has ratio
.9821494932 at 100M and passes. These are family quality comparisons, **not
X39 gate-cost lambda selection**. Archived model implementations were not
recertified and upstream conclusions were not rewritten.

The part-4 code, tests, receipts and documentation are included with this
partial result. A success title claiming four trained usage arms would be
false; the commit records the completed control and incomplete arms instead.
No compression claim follows from this control. The ordered X39 arm runs and
their registered poor/rich, decay, quality, story/depth/context and slope
receipts are still required.

## Part 4 partial result — two gates passed; training incomplete

`receipts/part4_decay.json` records a broker lease on physical card 6 and the
optimized X36 lane at fp32 decay **0.9474635124206543**. Against the CPU oracle,
the entire `freed`, `alloc_target`, and `case` tensors and final live mask agree
exactly, with nonzero decay deaths. This covers isolated births at B1, H4,
K64, V128, S2944, T512. It does **not** certify B64 training configurations,
gradients, ReplayBackward, or the full adversarial parity suite.

`receipts/part4_match.json` establishes the requested constructive CPU circuit.
With semantic keys and detached pre-read maximum scaled match evidence wired
to write-enable, novel enable/allocation fractions are **1.0/1.0** and repeat
fractions are **0.0/0.0**. Identity keys with the wire, and semantic keys without
the wire, both retain repeat enable/allocation **1.0/1.0**. Empty-store evidence
is zero; the hand-set construction uses write bias +2, match coefficient -1,
zero error coefficient, no POP, and an unchanged sequential CPU oracle. No
optimized read-to-gate implementation or learned behavior is claimed.

The CPU suite passes **11 tests**. Initial receipt execution caught a Python
scalar in the unwired ablation; replacing it with a shape-matched zero tensor
fixed the error before the passing receipt was generated.

**The requested training and evaluations remain unrun.** There is no lambda
operating point, per-arm parameter receipt, B64 oracle certification,
ReplayBackward attestation, trained slope, perplexity band, decay NLL curve,
or poor/rich deployment result. The B8 100M X34 checkpoint cannot supply the
registered exact-exposure control for B64 20M lambda selection. No supervisor
tool was exposed after discovery by name and description. No alternative
orchestration service was started. No success-titled commit or push was made
because that would claim that the cleared arms were trained.

Residual risks: hand-set score margins may not survive learned feature scales;
the architectural term is not yet implemented in an optimized training lane;
the existing error-magnitude coefficient belongs to the append-amplitude gate,
so a future implementation must explicitly specify its placement in the new
write-enable gate. Capacity-capped slopes can still conceal lost information.
All upstream edits already present in the shared tree remain untouched. No
`.pt` files were added. The sections below preserve earlier historical stops;
the part-4 instruction supersedes their blanket semantic stop for arms 1–3.

**The four-arm experiment is incomplete.** Pre-registration, CPU mechanism
counterexamples, and analysis of the existing X34 control are complete. No X39
training, lambda selection, parameter-match certification, GPU parity, or
trained-arm evaluation has run. This is not a scientific null from trained
regularized models.

## Measured control

The immutable X34 receipt already includes the requested longer contexts:

| T | Peak slots/head (all four heads) | Allocations/token/head | Write enabled | Pressure |
|---|---:|---:|---:|---:|
| 512 | 256 | .5 | 1 | 0 |
| 1024 | 512 | .5 | 1 | 0 |
| 2048 | 1024 | .5 | 1 | 0 |
| 4096 | 2048 | .5 | 1 | 0 |

OLS log-log peak slope is **1.0000000000000004** for every head. Story peak
median is 102, maximum 264. This supports the transcript diagnosis without
new training or GPU use. Mean occupancy and occupancy profiles are unavailable
in the original receipt and remain explicitly missing, rather than inferred
as fresh measurements. Source hashes are retained in `receipts/preflight.json`.

The actual imported T2 defaults give a **1,381,547.60-token** untouched-slot
fade horizon, longer than the task's approximate 200k estimate. The 128-token
arm's derived decay is **0.9474635256553754**, float32
**0.9474635124206543**, implied horizon **127.99996687** tokens. Exact GPU
death timing still needs configuration-specific parity; this calculation alone
does not certify the optimized lane.

## Blocking mechanism findings

1. `x30_tape_cell/naive_reference.py` selects append requests before computing
   the sigmoid append gate. The gate multiplies the stored value; allocation
   initializes strength to 1 regardless of its magnitude. X36 `kernels.py`
   uses the same order. CPU gate-logit extremes -20/+20 yield identical
   allocation traces with fixed nonzero values, while the read loss has a
   nonzero gate-logit gradient. Thus the differentiable path exists for value
   strength, not allocation rate. Indirect changes in residuals or POP may
   change future occupancy; that does not turn this gate into an allocation
   probability or supply a derivative of the hard allocation count.
2. Write correction is addressed by the age-1 slot. Random versus repeated
   keys produce identical routing/allocation traces when POP is disabled and
   content is fixed, although reads change. X37 W_in therefore cannot alone
   turn novelty routing into semantic matching. In addition,
   **R-ZERO-RESIDUAL appends**: zero-content repeated-concept counterexamples
   allocate on every token. Making this a no-op would change certified
   semantics. X36 explicitly treats cases 1, 2, and 3 as append requests.
3. X34 registration, CLI, and fingerprints specify effective batch **8**;
   the requested **64** changes optimizer updates. The control's existing
   receipt is at 100,003,840 tokens, not exactly the proposed 20M pilot
   exposures. Lambda selection needs an exact matched-exposure control
   validation receipt and a resolved batch policy.

These are scope decisions for the operator: define an allocating gate and its
gradient, define semantic match/no-op routing and recertify replay, and resolve
batch matching while preserving the instruction not to retrain control.
The requested Intercom contact_supervisor capability is unavailable: discovery
found no supervisor/intercom tool. No substitute service was started.

## Validation and remaining work

`/home/erikg/emender/.venv/bin/python -m
experiments.expressivity_tasks.x39_memory_usage.audit` passed and wrote the
source-hashed machine receipt. `/home/erikg/emender/.venv/bin/python -m pytest
experiments/expressivity_tasks/x39_memory_usage/tests -q` passed **5 tests**.
These are CPU oracle counterexamples and receipt-analysis checks; they are
explicitly not GPU oracle parity or ReplayBackward acceptance.

All upstream experiment directories were read-only in this task. No GPU was
allocated, no existing checkpoint was modified, and no .pt file was added.
The shared tree already contained extensive unrelated changes, including X36
source modifications, and those remain untouched. Source hashes identify the
working-tree implementations inspected; they must be checked again before any
future GPU certification.

After operator resolution, the implementation, broker runner, exact per-arm
parameter receipts, fresh oracle parity, ReplayBackward assertions, sequential
20M sweeps and 100M runs, poor/rich evaluation, occupancy profiles and decay
curves are all still required. Do not infer compression or acceptable quality
from this preflight. Rich S=2944 control does not saturate by T=4096, so a
zero-pressure result there would not demonstrate an improvement over control.

## Part 2 — redesigned CPU mechanisms (incomplete)

The operator superseded the earlier scope questions: use B64, penalize the
continuous **write-enable** sigmoid (not append amplitude), keep certified ISA
semantics, and combine semantic keys with that cost. The original preflight
above remains historical evidence, not a veto of these redesigned paths.

`redesigned.py` optimizes a learned scalar write bias initialized at +2, on
fixed features in [-1,1]. The objective is exactly lambda times mean sigmoid;
SGD uses learning rate 10 for 100 steps. The resulting sigmoid is thresholded
and detached as in the imported cell before entering the unmodified CPU oracle.
Costs 0/.001/.01/.1 yield physical allocation rates
.5/.5/.3359375/0. This proves indirect control across a hard threshold, not a
derivative of allocation count or language-model quality. The first nonzero
cost leaves allocation unchanged; monotonicity is non-increasing, not strict.
These artificial optimizer settings do not select a training lambda.

The derived decay arm is exercised for 512 tokens with S=512, showing decay
deaths and occupancy equilibrium without R-CAP. Full per-token/head occupancy
is retained in `receipts/redesigned_mechanisms.json`. The tests also verify the
strict .5 write-enable boundary. Eight CPU tests pass, including all five
original binding counterexamples.

**The requested training experiment is still incomplete.** No GPU lease, GPU
parity, ReplayBackward assertion, parameter certification, lambda sweep, 100M
training or trained-arm evaluation was performed. Poverty and semantic-repeat
mechanism receipts remain unimplemented. Pressure is returned as telemetry in
X34; its objective is cross entropy only. A poverty mechanism receipt must
therefore demonstrate an effect through capacity-limited reads on loss, rather
than treating pressure telemetry as an existing explicit loss penalty.
No claim is made that this absence alone prevents scarcity learning.

B64 rounds pilot/full runs to 611/3052 updates, 20,021,248/100,007,936 tokens.
This resolves the requested training batch but does not magically supply a B64
matched-exposure control for the 2% lambda decision: the original X34 checkpoint
is B8 at 100,003,840 tokens. That comparison remains unperformed.

Residual risks: synthetic gate closure can erase useful writes; the semantic
read-confirms circuit has not been learned or demonstrated; budget saturation
can produce misleading low occupancy slopes. X37's per-head W_in is not the
requested shared W_in and must not be reused unchanged. No upstream source or
ISA semantics was edited. No .pt files were added. Intercom supervisor tools
remain unavailable in this session. This is partial mechanism evidence only.

## Part 3 — poverty receipted; semantic gate failed, stopped before GPU

The ordered CPU preflight ran in `part3_mechanisms.py`, with source hashes and
full poverty occupancy traces in `receipts/part3_mechanisms.json`. **The semantic
mechanism did not pass. The fail-closed instruction stops this experiment before
GPU parity, training, or evaluation of any arm.** No GPU broker was invoked.

Poverty uses 24 orthogonal bindings, one enabled write every other token, then
24 read-only recall queries, identically across four heads. Only budget changes.
The real X30 CPU cell's 128-token cap/request pressure register is recorded;
its read outputs agree with the differentiable sequential oracle within 2e-5.
Cross entropy is computed from 20 times the recalled value with the binding
identity as target. Values receive nonzero loss gradients. There is no added
pressure penalty and these are synthetic recall losses, not LM perplexities.

| S/head | Peak live/head | R-CAP (all heads) | Pressure max | Recall CE | Recall accuracy |
|---|---|---|---|---|---|
| 8 | 8 | 64 | .666667 | 3.882920 | .333333 |
| 16 | 16 | 32 | .333333 | 2.166516 | .666667 |
| 32 (rich counterfactual) | 24 | 0 | 0 | 0 (fp32) | 1 |

This receipts a scarcity teaching signal through capacity-limited reads and
loss, not successful scarcity learning or transfer to S=2944.

Semantic repeats use eight concepts, each presented twice with different alias
identities. A single hand-set shared W_in folds the alias pairs to common
content keys before identical per-head identity geometry. Mean target read on
repeat tokens improves from .090664 (identity keys) to .999920 (semantic keys).
However, repeat write-enable and allocation fractions are **1.0 in both cases**;
allocation traces are exactly identical. Retrieval succeeds; gate closure fails.

The inspected X36/X37 path computes `W_write(x)` before the tape scan and has no
current tape-read input to that gate. Changing shared key/query features does
not itself implement read-confirms-stored gating. This witness does not prove
that transformer context cannot learn repetition indirectly; it records that
the requested CPU circuit was not established by the hand-set construction.
No read-to-gate architectural redesign was invented, and no trained semantic
null is claimed. The arm requires redesign or disposition before GPU work.

Validation: the full X39 CPU test suite passes **10 tests**, including two new
mechanism regression tests. A passing regression asserts the semantic failure
and stop condition; it does not turn the scientific gate green. The Intercom
supervisor tool remains unavailable after discovery by name and description.
Upstream directories and unrelated working-tree changes were left untouched.
No .pt files were added. The requested success commit title would misstate the
outcome; the part-3 commit records the mechanism failure instead.

Residual work: decay GPU death-schedule parity, per-arm parameter receipts,
ReplayBackward assertions, lambda operating-point selection with a valid
matched-exposure control, all training, and every trained-arm evaluation remain
unrun. The prior B8-control/B64-training comparison limitation remains.
