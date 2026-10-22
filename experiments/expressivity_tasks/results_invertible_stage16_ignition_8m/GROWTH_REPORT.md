# X16 unified ignition-sweep growth report — four terminal FAIL arms

## Measured threshold result

**E97's observed ignition window is (1/32, 1/16]; GDN2's is
(1/512, 1/136].** All four X16 sweep arms completed the fixed T16 500K
ceiling without ANY position reaching the registered ignition criterion
>=50/512. These are completed no-ignition **FAIL** results, not the original
infrastructure-censored observations and not claims of impossibility beyond500K.

Together with existing receipts — both architectures fail at w=0 [X14], E97
ignites at w=1/16 [X9/X10], and GDN2 ignites at w=1/136 [X8] — these results
confirm a nonzero ignition bracket for both architectures: **same cliff shape,
different scaffold share**. The cliff description is the observed failed-dose /
successful-dose contrast, not a resolved response curve. The brackets are
**confirmed, not closed**: no interior point in either remaining open interval
has been tested; no exact threshold location, interpolated success, or assumed
monotonicity is claimed.

The roughly **factor 4–8 scaffold-share separation** is a historical-dose
comparison, not a measured threshold-ratio bound: GDN2's receipted successful
1/136 is 4.25x below E97's failed1/32 and 8.5x below its successful1/16.
The successful-dose endpoints alone differ by8.5x (roughly8x). If one summarizes
the windows by a single threshold per architecture, their ratio is bounded only
roughly4–32, not4–8; that summary does not establish a unique monotone threshold.
X8 differs in optimizer and non-P1 weighting, so this is not a matched causal
estimate of architecture-specific sensitivity to w.

|Architecture / explicit P1 share w|Fixed-final verdict|P1 final / max /512|P2+ largest max /512|Any-position first50|Final validation CE|
|---|---|---|---|---|---|
|E97, 1/64 = .015625|FAIL, complete500K|11 / 20|3|censored@500000|7.971306|
|E97, 1/32 = .03125|FAIL, complete500K|6 / 19|3|censored@500000|7.968394|
|GDN2, 1/512 = .001953125|FAIL, complete500K|0 / 2|3|censored@500000|8.221836|
|GDN2, 1/2048 = .00048828125|FAIL, complete500K|0 / 2|3|censored@500000|8.234554|

## Registration and scope

[REGISTRATION.md](REGISTRATION.md), Measurement1 predictions **verbatim**:

> (a) E97 ignition bracket lies between existing w=0 (X14 no ignition) and w=1/16
> (X10 ignition); the two interior points measure, not assume, that bracket.
> (b) GDN2 failure point, if in the ladder, is measured at1/512 or1/2048. X8's1/136
> ignition is CONTEXT ONLY: its optimizer and non-P1 weighting differ; no matched
> causal attribution to explicit w across that historical endpoint.

Applied as registered: **both predicted threshold existence, not a particular
untested interior outcome**. Prediction(a)'s two originally interior E97 doses
both fail; its empirical bracket narrows to(1/32,1/16]. Prediction(b)'s conditional
failure point is observed at BOTH1/512 and1/2048;1/512 is the tighter failing
endpoint against historical1/136. Thus brackets are confirmed, not closed.
Prediction(c)'s position-level first50/508/512, maxima and fixed-final readouts
are preserved below; P1 is non-composition, P2–P16 are composition.

This is standalone lambda01 expressivity, not Frontier or resilient DiLoCo.
All four arms are **seed0 only**, scratch fixed-T16, frozen512 validation
examples, evaluation every500 updates including entry, ceiling500000.
No multi-seed generality, exact critical dose, or eventual ignition claim follows.
The shared recipe is X10 AdamW+cosine, dimension256/four layers/batch32,
FP32 parameters/BF16 autocast; E97 uses legacy-state/sequential Triton evaluation
and GDN2 stock BF16 FLA chunk recurrence. The training objective is
w*CE(P1)+(1-w)*mean CE(P2..P16), with full targets/masks unchanged. Reported
validation CE is the **unweighted all16-position mean**, not that weighted
training objective; it must not be compared as though the two were identical.

### Historical endpoint receipts (context, not new X16 arms)

Paths in this table are relative to `experiments/expressivity_tasks/`.
First50 times are sampled validation updates, not exact continuous-time onset.

|Receipt|P1 share|Observed ignition / endpoint|Source|
|---|---|---|---|
|X14 E97 / GDN2 seed0|0|Neither ever>=50 through500K|`results_invertible_stage14_nofreebie_8m/{e97_seed0,gdn2_seed0}/validation.json`; [X14 report](../results_invertible_stage14_nofreebie_8m/GROWTH_REPORT.md)|
|X9 E97 seed0|1/16|Any-position first50=47000; final P1–P3=512 each|`results_invertible_stage9_anneal_8m/e97_bf16_seed0/{validation,result}.json`|
|X10 E97 AdamW seed0|1/16|Any-position first50=24000; final P1–P4=512 each|`results_invertible_stage10_optimizers_8m/adamw_seed0/{validation,result}.json`|
|X8 GDN2 seed0|1/136|Any-position first50=139500; final P1/P2=505/511|`results_invertible_stage8_paydepth_8m/gdn2_bf16_seed0/{validation,result}.json`|

X9 is annealed historical context; X10 is the registered shared optimizer recipe.
X8's depth-proportional weighting and optimizer differ from this sweep. No
historical T64 failure is treated as a T16 dose point.

## Loss signatures: exact-marginal versus P1 extraction

The exact uniform-marginal CE reference is ln3721=8.221747728;
joint chance is1/3721 (expected0.1376/512 per position). “Exact-marginal” names
this reference/signature, not literal equality of every measured CE or proof
that absolutely nothing was learned. Counts and position/coordinate losses
are needed alongside aggregate CE.

|Arm|Final CE minus ln3721|P1 final CE|Mean P2+ final CE|P1 x / y correct out of512|Signature|
|---|---|---|---|---|---|
|E97 1/64|-0.250442|4.170705|8.224679|512 / 11|Below-marginal P1 x-extraction; no material P1 or composition ignition|
|E97 1/32|-0.253354|4.141720|8.223505|512 / 6|Below-marginal P1 x-extraction; no material P1 or composition ignition|
|GDN2 1/512|+0.000088|8.221397|8.221865|13 / 8|Exact-marginal signature (aggregate8.2218), no material ignition|
|GDN2 1/2048|+0.012807|8.243686|8.233946|9 / 7|Near/slightly above exact-marginal reference (aggregate8.2346), no material ignition|

E97 learns P1's x coordinate perfectly but not the full joint P1 map: its final
P1 CE is close to ln61=4.110874, while P2+ stays near ln3721. Its lower aggregate
CE is consequently **not composition learning**, nor does P1's max20 or19 cross50.
GDN2 shows neither this P1-extraction signature nor material composition at these
doses. Both architectures therefore fail the SAME ignition readout, despite
qualitatively different below-threshold loss signatures.

|Update|E97 1/64 CE|E97 1/32 CE|GDN2 1/512 CE|GDN2 1/2048 CE|
|---|---|---|---|---|
|5000|8.336712|8.340329|8.407546|8.377156|
|50000|8.221631|8.221687|8.221878|8.221844|
|100000|8.222083|8.221795|8.221973|8.222036|
|250000|8.221871|8.221802|8.221732|8.222164|
|500000|7.971306|7.968394|8.221836|8.234554|

These sampled loss curves are descriptive; the fixed-final500K counts remain
primary. Full500-step validation series are in each authoritative arm path below.

## Infrastructure provenance — split admissions, not a blanket resume claim

The approximately21:00 UTC2026-10-21 ENOSPC interruption was infrastructure,
not architecture failure. Original interrupted outputs remain censored and
immutable. The [resume recovery report](infra_resume/RECOVERY_REPORT.md) and
[restart recovery report](infra_restart/RECOVERY_REPORT.md) retain the amendments
and admission tests. The four terminal scientific arms use the following paths:

|Arm|Authoritative output namespace|Admission / checkpoint|Historical tail treatment|
|---|---|---|---|
|E97 1/64|`infra_resume/e97_w64_seed0/`|REPLAY-VERIFIED; checkpoint244000|106 full updates matched exactly through244106; partial244107 had no readable numeric fields|
|E97 1/32|`infra_restart/e97_w32_seed0/`|APPROVED CHECKPOINT RESTART; checkpoint245000|Historical245001..245475 and partial245476 superseded by re-execution; NOT replay-verified|
|GDN2 1/512|`infra_resume/gdn2_w512_seed0/`|REPLAY-VERIFIED; checkpoint129000|920 full updates matched exactly through129920; partial129921 had no readable numeric fields|
|GDN2 1/2048|`infra_resume/gdn2_w2048_seed0/`|REPLAY-VERIFIED; checkpoint130000|148 full updates matched exactly through130148; partial130149 had no readable numeric fields|

Each retained checkpoint restored model, AdamW moments and NumPy/torch/CUDA RNG.
For the three verified arms, exact diagnostic replay is evidence, **not proof
of bitwise full-state trajectory identity**; an empty partial-row diagnostic set
is not a full-update comparison. Their `replay_verified.json` and replay-confirmed
amendments preserve that limit.

E97 1/32's exact replay aborted at245104: grad_norm
0.03506257385015488 versus0.03506257012486458. Its
[separate approved restart amendment](infra_restart/AMENDMENT-2026-10-21-restart-e97_w32_seed0.md)
and `restart_admitted.json` explicitly supersede the historical post-checkpoint
tail; no at-death restoration, exact replay success, or trajectory identity is
claimed. Failed exact-gate evidence stays in `infra_resume/e97_w32_seed0/`.
Floating-point nondeterminism is plausible, not an isolated proven cause; no
tolerance was relaxed. Original global schedules and500K ceilings continued
without warmup reset, with original evaluation/readout rules. The restarted
arm is not an uninterrupted matched-seed trajectory.

Every authoritative `result.json` says `status=complete`, `total_steps=500000`,
`stop_reason=safety_ceiling`, with final step500000. Every corresponding
`<namespace>/receipts/<arm>/terminal.json` has exit_code0, status complete, and
binds the result SHA256. “Complete” is operational success; **FAIL** here is
scientific no-ignition at the fixed ceiling. The old root-arm infrastructure
terminals must not override these admitted continuations.

## Full positional tables

All counts are joint correct out of512. Final means fixed update500000;
max is over the complete admitted validation history, including entry and
retained pre-checkpoint evaluations. Every first50/508/512 is **right-censored
at500000**, not an observed zero-time crossing. Every arm has1001 evaluations
at steps0,500,...,500000. P1 is the non-composition scaffold; P2–P16 are
composition. No position in any sweep arm ever reaches50/512.

### E97 w=1/64 — e97_w64_seed0

Source: [infra_resume/e97_w64_seed0/validation.json](infra_resume/e97_w64_seed0/validation.json);
[infra_resume/e97_w64_seed0/result.json](infra_resume/e97_w64_seed0/result.json).

|Position / role|Final /512|Max /512|First50|First508|First512|Final CE|
|---|---|---|---|---|---|---|
|P1 non-composition|11|20|censored@500000|censored@500000|censored@500000|4.170705|
|P2 composition|0|2|censored@500000|censored@500000|censored@500000|8.229189|
|P3 composition|0|2|censored@500000|censored@500000|censored@500000|8.224626|
|P4 composition|0|3|censored@500000|censored@500000|censored@500000|8.223338|
|P5 composition|0|3|censored@500000|censored@500000|censored@500000|8.226165|
|P6 composition|0|2|censored@500000|censored@500000|censored@500000|8.228269|
|P7 composition|0|2|censored@500000|censored@500000|censored@500000|8.226319|
|P8 composition|0|2|censored@500000|censored@500000|censored@500000|8.220571|
|P9 composition|0|2|censored@500000|censored@500000|censored@500000|8.221124|
|P10 composition|0|2|censored@500000|censored@500000|censored@500000|8.220147|
|P11 composition|0|2|censored@500000|censored@500000|censored@500000|8.224387|
|P12 composition|0|3|censored@500000|censored@500000|censored@500000|8.231762|
|P13 composition|0|2|censored@500000|censored@500000|censored@500000|8.224523|
|P14 composition|0|3|censored@500000|censored@500000|censored@500000|8.224766|
|P15 composition|0|2|censored@500000|censored@500000|censored@500000|8.226555|
|P16 composition|1|2|censored@500000|censored@500000|censored@500000|8.218443|

### E97 w=1/32 — e97_w32_seed0

Source: [infra_restart/e97_w32_seed0/validation.json](infra_restart/e97_w32_seed0/validation.json);
[infra_restart/e97_w32_seed0/result.json](infra_restart/e97_w32_seed0/result.json).

|Position / role|Final /512|Max /512|First50|First508|First512|Final CE|
|---|---|---|---|---|---|---|
|P1 non-composition|6|19|censored@500000|censored@500000|censored@500000|4.141720|
|P2 composition|0|2|censored@500000|censored@500000|censored@500000|8.227759|
|P3 composition|0|2|censored@500000|censored@500000|censored@500000|8.223144|
|P4 composition|0|3|censored@500000|censored@500000|censored@500000|8.222199|
|P5 composition|0|2|censored@500000|censored@500000|censored@500000|8.224263|
|P6 composition|0|2|censored@500000|censored@500000|censored@500000|8.223610|
|P7 composition|0|2|censored@500000|censored@500000|censored@500000|8.226247|
|P8 composition|1|3|censored@500000|censored@500000|censored@500000|8.223082|
|P9 composition|2|2|censored@500000|censored@500000|censored@500000|8.224302|
|P10 composition|0|2|censored@500000|censored@500000|censored@500000|8.222206|
|P11 composition|0|2|censored@500000|censored@500000|censored@500000|8.222493|
|P12 composition|0|3|censored@500000|censored@500000|censored@500000|8.224626|
|P13 composition|1|3|censored@500000|censored@500000|censored@500000|8.224929|
|P14 composition|0|3|censored@500000|censored@500000|censored@500000|8.221616|
|P15 composition|0|3|censored@500000|censored@500000|censored@500000|8.221146|
|P16 composition|0|2|censored@500000|censored@500000|censored@500000|8.220954|

### GDN2 w=1/512 — gdn2_w512_seed0

Source: [infra_resume/gdn2_w512_seed0/validation.json](infra_resume/gdn2_w512_seed0/validation.json);
[infra_resume/gdn2_w512_seed0/result.json](infra_resume/gdn2_w512_seed0/result.json).

|Position / role|Final /512|Max /512|First50|First508|First512|Final CE|
|---|---|---|---|---|---|---|
|P1 non-composition|0|2|censored@500000|censored@500000|censored@500000|8.221397|
|P2 composition|0|2|censored@500000|censored@500000|censored@500000|8.223747|
|P3 composition|0|2|censored@500000|censored@500000|censored@500000|8.221162|
|P4 composition|0|2|censored@500000|censored@500000|censored@500000|8.219420|
|P5 composition|0|2|censored@500000|censored@500000|censored@500000|8.221324|
|P6 composition|0|2|censored@500000|censored@500000|censored@500000|8.223269|
|P7 composition|0|2|censored@500000|censored@500000|censored@500000|8.222807|
|P8 composition|0|2|censored@500000|censored@500000|censored@500000|8.221458|
|P9 composition|1|2|censored@500000|censored@500000|censored@500000|8.220783|
|P10 composition|0|2|censored@500000|censored@500000|censored@500000|8.220355|
|P11 composition|0|2|censored@500000|censored@500000|censored@500000|8.222678|
|P12 composition|0|3|censored@500000|censored@500000|censored@500000|8.221793|
|P13 composition|0|2|censored@500000|censored@500000|censored@500000|8.223178|
|P14 composition|0|3|censored@500000|censored@500000|censored@500000|8.222910|
|P15 composition|0|2|censored@500000|censored@500000|censored@500000|8.222029|
|P16 composition|1|2|censored@500000|censored@500000|censored@500000|8.221054|

### GDN2 w=1/2048 — gdn2_w2048_seed0

Source: [infra_resume/gdn2_w2048_seed0/validation.json](infra_resume/gdn2_w2048_seed0/validation.json);
[infra_resume/gdn2_w2048_seed0/result.json](infra_resume/gdn2_w2048_seed0/result.json).

|Position / role|Final /512|Max /512|First50|First508|First512|Final CE|
|---|---|---|---|---|---|---|
|P1 non-composition|0|2|censored@500000|censored@500000|censored@500000|8.243686|
|P2 composition|0|3|censored@500000|censored@500000|censored@500000|8.235845|
|P3 composition|0|2|censored@500000|censored@500000|censored@500000|8.242181|
|P4 composition|0|2|censored@500000|censored@500000|censored@500000|8.236140|
|P5 composition|0|3|censored@500000|censored@500000|censored@500000|8.239259|
|P6 composition|0|2|censored@500000|censored@500000|censored@500000|8.246490|
|P7 composition|1|2|censored@500000|censored@500000|censored@500000|8.245062|
|P8 composition|0|2|censored@500000|censored@500000|censored@500000|8.221105|
|P9 composition|0|2|censored@500000|censored@500000|censored@500000|8.227242|
|P10 composition|0|3|censored@500000|censored@500000|censored@500000|8.228775|
|P11 composition|0|2|censored@500000|censored@500000|censored@500000|8.224703|
|P12 composition|0|2|censored@500000|censored@500000|censored@500000|8.233759|
|P13 composition|0|2|censored@500000|censored@500000|censored@500000|8.229623|
|P14 composition|0|2|censored@500000|censored@500000|censored@500000|8.227296|
|P15 composition|1|2|censored@500000|censored@500000|censored@500000|8.234263|
|P16 composition|0|2|censored@500000|censored@500000|censored@500000|8.237442|

## Terminal evidence hashes

SHA256 values verified against the local terminal artifacts; result hashes also
match the per-arm terminal receipts. Artifact paths are relative to this report.

|Arm path|result.json SHA256|validation.json SHA256|
|---|---|---|
|`infra_resume/e97_w64_seed0`|`62085a1fcd474463796d04d858877c92bb2329257e1e9fb83066780e4cdf85fb`|`c2f96f3c2f671bb9b28b7a57eb09104ee355e838c3c7806b4d2799d3008182ad`|
|`infra_restart/e97_w32_seed0`|`073962a273b0d4c2fc683d6b7d5a68b1773746974abc67a4a3a7e332e9592f7a`|`e591285ea83c0fc0726b64d402b6440bf30856609850137975c65f0d34cd7813`|
|`infra_resume/gdn2_w512_seed0`|`acf3be17a77743c845fa6f83157e574fee222aa2cadc49edfc2185e89c6fdba6`|`921f3ef9aabd778819b11bf0ff260b2a769b193023f0ad8d6fbc72bc00c807a0`|
|`infra_resume/gdn2_w2048_seed0`|`025031cd5fe23cf593037facf9d445b49316f4b77374c36563dae53526addf48`|`7323f8f04280f62a64a767436bbff33643bb215370d14cce3cbf6a7f1f3f4bf8`|

## Curriculum cross-reference only — X16E closed

The curriculum/extension result is reported separately in the closed
[X16E growth report](curriculum_extension/GROWTH_REPORT.md), with its own
[amendment](curriculum_extension/AMENDMENT.md). It is not a fifth ignition-dose
arm, not an interior-point test, and not evidence that closes either sweep
bracket. No curriculum score or prediction is pooled into this sweep verdict.

## Corrections and interpretation limits

- **Concession #14:** the earlier “factor4–8” shorthand compared historical
  scaffold-share endpoints, not measured threshold ratios. The successful doses
  1/16 versus1/136 differ by8.5x; the lower4.25x comparison uses E97's FAILED1/32.
  The windows alone permit only a roughly4–32 threshold-ratio bound under a
  single-threshold reading. Neither architecture's interior threshold is located.
- The old root growth report's infrastructure-censored FAIL wording is superseded
  for these four arms by terminal500K no-ignition. Infrastructure interruption
  itself remains non-scientific failure; recovery paths are disclosed per arm.
- No monotonicity, seed replication, optimizer-isolated cross-architecture
  causality, exact threshold, or untested interior success/failure is asserted.
  E97's below-marginal P1 extraction is not composition; GDN2's near-marginal
  signature alone is not proof of no learning. The registered count test decides.

## Validation

On lambda01, using the existing Python3.12 venv (not Frontier): checked all four
terminal result hashes, exit/status/ceiling/final-step fields,1001 unique scheduled
evaluations per arm, all64 final/max/first50/508/512 positional rows and final CEs,
loss-signature/trajectory summaries, historical first50/endpoints, and the
three-replay-verified / one-checkpoint-restart admission split. Markdown relative
links and whitespace are checked. This report changes no training, registration,
stopping criterion, or executable code; no new training or bracket-closing arm
was run and no automated test source was changed.
