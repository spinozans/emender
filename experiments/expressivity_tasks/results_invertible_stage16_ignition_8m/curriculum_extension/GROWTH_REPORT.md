# X16E growth report — TRANSFER, retention ENHANCED

## Fixed-final verdict against the registration

The completed seed0 extension reaches **T64 P1–P5
[512,512,512,512,10] /512** at extension update500000, after500000
teaching updates (1M total). The final T16 retention prefix is
**[512,512,512,512,6] /512: ENHANCED**, not merely preserved. Both horizons
end with a contiguous four-perfect prefix. Fixed-final500K is primary;
transient maxima and crossings are descriptive, sampled every500 updates.

Under [AMENDMENT.md](AMENDMENT.md), registered prediction **(a) TRANSFER is
supported at the fixed-final endpoint**: T64 P2+ exceeds50/512, taught T16
P1–P3 each retain512/512 (criterion >=508), and P4 is NEW material acquisition,
not just inherited zero-shot performance. T64 step0 already has P1–P3 perfect
and P4=9; its material position set (>=50) is {P1,P2,P3}. Maximum and final
material sets are both {P1,P2,P3,P4}; new transient and final sets are both
{P4}. The material frontier therefore advances3->4. P1 is non-composition;
P2–P4 are the composition evidence. Prediction(b)'s zero/chance NO-TRANSFER
outcome, and the separate transfer-without-new-acquisition outcome, do not hold.

Registered prediction **(c) RETENTION DAMAGE has a transient-only reading,
not a fixed-final destruction verdict**. Taught P1/P2/P3 minima are508/451/429;
P2 first falls below508 at3000 and P3 at500. All three recover to512 at the
endpoint. T16 P4 improves5->512 without T16 replay: retention is ENHANCED.
The taught P4=5 checkpoint ultimately yields P4=512 at BOTH horizons; the
actual T64 step0 count is9, not5, because its frozen validation pool differs.
The full16-position baseline/final/minimum table below preserves transient
versus final damage; exact trajectories are in `e97_extension_seed0/validation.json`.

**P5 is budding, not endpoint growth of the material frontier.** Its T64
maximum is17 and final10; T16 final is6. Neither horizon ever reaches50 for
P5. P5 first-50/508/512 are right-censored at500000 extension updates; no
fifth learned position, deeper endpoint, or eventual acquisition is claimed.

## Teach-phase history and restart caveat

The original [X16 registration](../REGISTRATION.md) required >=4 positions
>=508/512 (including P1) before T64. Teaching hit its500K ceiling with only
P1–P3 perfect: [512,512,512,5,...]. P4 peaked at11, never reached50, and
stalled at5. The original run terminated `criterion_not_reached`; its T64
phase was censored, not a T64 failure or a successful criterion transition.
X16E is a separately authorized amendment despite that miss, NOT a weakened
or retrospectively satisfied original criterion. The task shorthand P4=8 is
not the hash-bound teaching result;5 is authoritative.

Teaching carries restart divergence at update252204: grad_norm
0.09198356419801712 versus0.09198416024446487. Preserve BOTH registered
readings: (1) the checkpoint restart may have damaged late-budding P4 relative
to the same-seed four-perfect X10 AdamW trajectory; (2) the divergence may be
irrelevant and this teaching trajectory simply stalled. Neither causal reading
is established by these receipts, and extension success does not resolve them.

The extension restored terminal model, AdamW moments and RNG state, then used
T64-only uniform loss, no T16 replay/mixing, and a fresh warmup-free cosine
3e-4->3e-6 for500K updates. This is not the original attained-LR schedule;
the intervention bundles staging, horizon and the registered schedule change.

## Acquisition and early retention timings

All times below are EXTENSION updates, not global teaching-plus-extension time.

|Readout|Receipt|Interpretation|
|---|---|---|
|P2 at2000|510/512 on T64 AND T16|Early retention, NOT first acquisition: both started at512|
|T64 P4 first-50|56000 (56K)|First NEW material position beyond the step0 set|
|T64 P4 first-250|168000 (168K)|Intermediate acquisition milestone|
|T64 P4 first-508|337500 (337.5K)|Near-perfect crossing|
|T64 P4 first-512|357500 (357.5K)|First perfect P4, distinct from final512|
|T16 P4 first-50 /250 /508 /512|54500 /163500 /336500 /355500|Enhancement on the independent retention pool|

## Causal contrast — historical endpoints, bounded interpretation

|Contrast|Cold-start T64 receipts at1M|Staged seed0 at500K teach +500K extend|What this supports / does not establish|
|---|---|---|---|
|Same-seed0 anchor, seed1 context|X12 seed0: zero-perfect, P1–P5 [0,0,0,0,0], maximum any position3; X13 seed1: zero-perfect, P1–P5 [0,0,0,0,0], maximum any position4. Both dead through1M under the >=50 criterion.|Same seed0: four-perfect, P1–P5 [512,512,512,512,10].|A staged route escapes the observed same-seed cold-start failure at equal total update count; not an isolated horizon-only or schedule-only effect.|
|Lottery escape versus prescribed staging|X13 seed2 escapes: two-perfect + P3=309 at1M, P1–P5 [512,512,309,1,0].|Deterministic staged intervention: four-perfect at1M total.|Staging reaches further than this observed lottery escape; “deterministic” describes the prescribed seeded procedure, NOT proven repeatability or all-seed reliability.|

Sources: [X12 registration](../../results_invertible_stage12_separation_8m/REGISTRATION.md),
X12 `e97_bf16_seed0/validation.json`, and
[X13 growth receipt](../../results_invertible_stage13_zerocheck_8m/GROWTH_REPORT.md)
with its seed1/seed2 validation histories. These are historical endpoint
contrasts, not a new randomized matched ablation: cold T64 used its registered
1M cosine, while X16E used two phases and a fresh extension cosine. Cold zeros
are bounded observations, not proof of architectural impossibility; seed2
already rules out a universal cold-start-zero claim.

## Evidence and scope

The completed `e97_extension_seed0/result.json` reports `status=complete`,
`teach_steps=500000`, `t64_steps=500000`, and `stop_reason=safety_ceiling`.
The worker terminal receipt has exit0 and binds that result to SHA256
`07045c9d6a85956926e187c4f3901f61bc774fa9c39624e67bc8b6c0beb7735a`;
`receipts/watcher/terminal.json` records exit0, “verified fixed-final500K”.
There are1001 paired T64/T16 validations (entry plus every500 through500K).
The source teaching checkpoint SHA256 is
`870c2dcf5e863dcab817e3c6b0150d054e91cff8456d231378d3748676e06b1f`;
the final extension checkpoint SHA256 is
`bb09d1eef9e3aa1d942c5efe1d36ca1bd9e6b00911d1aefd6716faedea1c4200`.
These are recorded provenance receipts, not a claim of a fresh checkpoint replay.

**Scope: one seed, one curriculum design, only the completed receipts.**
This supports learning transfer and endpoint retention-enhancement for this
staged recipe; it does not isolate a mechanism, prove deterministic generality,
resolve restart causality, establish deeper-than-four growth, or predict P5's
future. Standalone lambda01 expressivity only: no Frontier, Slurm, resilient
DiLoCo or system-scale conclusion. No additional training was run for this report.

## T64 full positional receipt

Observed through extension update500000. None = crossing right-censored there.
|Position|T64 baseline|Latest|Max|First50|First508|First512|
|---|---|---|---|---|---|---|
|P1|512|512|512|0|0|0|
|P2|512|512|512|0|0|0|
|P3|512|512|512|0|0|0|
|P4|9|512|512|56000|337500|357500|
|P5|0|10|17|None|None|None|
|P6|0|0|2|None|None|None|
|P7|0|0|2|None|None|None|
|P8|0|0|2|None|None|None|
|P9|0|0|3|None|None|None|
|P10|0|0|2|None|None|None|
|P11|0|0|3|None|None|None|
|P12|1|0|2|None|None|None|
|P13|0|0|2|None|None|None|
|P14|0|0|2|None|None|None|
|P15|1|0|2|None|None|None|
|P16|0|0|2|None|None|None|
|P17|0|0|2|None|None|None|
|P18|1|3|3|None|None|None|
|P19|0|0|2|None|None|None|
|P20|0|1|3|None|None|None|
|P21|0|0|2|None|None|None|
|P22|0|0|2|None|None|None|
|P23|0|0|3|None|None|None|
|P24|0|0|3|None|None|None|
|P25|0|0|3|None|None|None|
|P26|1|0|2|None|None|None|
|P27|0|1|2|None|None|None|
|P28|0|0|2|None|None|None|
|P29|0|0|2|None|None|None|
|P30|0|0|2|None|None|None|
|P31|1|0|2|None|None|None|
|P32|1|1|3|None|None|None|
|P33|0|0|2|None|None|None|
|P34|0|0|2|None|None|None|
|P35|1|0|3|None|None|None|
|P36|1|0|2|None|None|None|
|P37|0|0|2|None|None|None|
|P38|0|0|2|None|None|None|
|P39|0|0|2|None|None|None|
|P40|0|0|2|None|None|None|
|P41|1|0|2|None|None|None|
|P42|1|0|2|None|None|None|
|P43|0|0|2|None|None|None|
|P44|0|1|2|None|None|None|
|P45|0|0|2|None|None|None|
|P46|0|0|3|None|None|None|
|P47|1|0|2|None|None|None|
|P48|0|0|3|None|None|None|
|P49|0|0|2|None|None|None|
|P50|0|0|2|None|None|None|
|P51|0|0|3|None|None|None|
|P52|0|0|2|None|None|None|
|P53|0|0|2|None|None|None|
|P54|0|1|2|None|None|None|
|P55|1|0|2|None|None|None|
|P56|0|0|3|None|None|None|
|P57|0|0|3|None|None|None|
|P58|0|0|3|None|None|None|
|P59|0|0|3|None|None|None|
|P60|0|0|2|None|None|None|
|P61|0|0|2|None|None|None|
|P62|0|0|2|None|None|None|
|P63|0|0|2|None|None|None|
|P64|1|0|2|None|None|None|

## retention_t16 (exact full trajectories in validation.json)

P1–P3 are the taught positions. P4–P16 were untaught at transition; their
below508-at0 entries are not damage to a learned position. “Latest drop” is
baseline minus final (negative means improvement); minima are independent
per-position minima, not necessarily one jointly observed validation.

|Position|Baseline|Latest|Min|Latest drop|First below508|
|---|---|---|---|---|---|
|P1|512|512|508|0|None|
|P2|512|512|451|0|3000|
|P3|512|512|429|0|500|
|P4|5|512|0|-507|0|
|P5|0|6|0|-6|0|
|P6|1|0|0|1|0|
|P7|0|0|0|0|0|
|P8|0|0|0|0|0|
|P9|0|0|0|0|0|
|P10|0|0|0|0|0|
|P11|0|0|0|0|0|
|P12|0|1|0|-1|0|
|P13|0|1|0|-1|0|
|P14|0|0|0|0|0|
|P15|0|0|0|0|0|
|P16|0|0|0|0|0|
