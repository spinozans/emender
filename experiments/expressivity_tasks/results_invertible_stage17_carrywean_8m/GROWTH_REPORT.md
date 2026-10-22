# X17 terminal carry-wean growth — WEAN-TRANSFER FIRES at T64

## Fixed-final verdict and registered criteria

The completed seed0 run ends at **1,000,000 updates with training
`carry_probability = 0.0` exactly**, not merely near zero. Final P1–P5 are
**[512,512,301,20,0] /512**: two-perfect plus P3=58.7890625%.
Final and maximum single-validation perfect-prefix depths are both2.
**Every evaluation has `eval_carry_probability = 0.0`**: all carry slots are
neutral, including evaluations while the training crutch is nonzero.
This is learned-with-crutch, retained-without at the fixed endpoint, not
true-feedback evaluation and not a post-hoc best-checkpoint verdict.

The immutable [REGISTRATION.md](REGISTRATION.md) states, verbatim:

> (a) WEAN-TRANSFER: the run acquires carry positions materially (any P2+ >=50/512 during or by end) with the crutch annealed to zero - examples-of-working training transfers to the no-crutch regime at the horizon where cold learning fails; the fix works and the paper gets its training-technology result.

> (b) CRUTCH-DEPENDENT-AT-DEPTH: composition appears while p is high but decays as p anneals (X6's cycling signature, flattened) - the carry remains crutch-bound at T64 and the wean problem persists at long horizons.

> (c) The P1 diagnostic rides free: does the supervised carry slot also ignite P1-like visible learning earlier than Muon/AdamW cold runs? Logged either way.

**Criteria application: (a) WEAN-TRANSFER FIRES: P2+ >=50/512 with the crutch
at zero.** At fixed-final1M, training p=0.0 and neutral-eval P2=512 and
P3=301, both >=50. The early crossing alone would NOT suffice; survival at
zero is separately observed here. P1 is the extraction/non-composition
control, not the composition evidence. Material positions at final are
{P1,P2,P3}; composition positions are {P2,P3}.

**(b) Persistent crutch-bound decay is not the terminal outcome for the
learned positions.** There are real transient dips (below), not monotonic
retention. P2 ultimately holds512 and P3 grows to its maximum301 at p=0.
**(c) P1 ignition is observed** at first-50=29K, first-perfect=53.5K,
without a supervised slot before P1. Historical cold X12 AdamW T64 seed0
never reaches50 in1M; cold X15G Muon@3x T64 seed0 peaks at19 in1M.
This is earlier visible learning than those failed cold trajectories,
not an isolated causal effect of a P1 slot or a universal optimizer ranking.
The older cold X10 Muon T16 first-50=19K / first-perfect=100K is at a
*different horizon and schedule*: X17 is later to50 but earlier to512 there.

## Wean trajectory: crutch level at every acquisition milestone

One global schedule, unchanged across the restart:
`p(step) = 0.5 * (1 - step / 1000000)`. Training logs p **every update**;
the values below were checked against those rows AND validation-row p.
Displayed p is rounded to six decimals; final p is literally JSON `0.0`.
There is no re-crutch/reset and no adaptive schedule. True previous-state
slots occur only before P2–P64; P1 has no extra slot. All counts are joint-state
correct/512 on the frozen T64 pool; validation cadence is500 updates, so these
are first **observed** crossings, not unsampled continuous-time events.

|Position|Crossing|First update|Count at crossing|Training carry_p|Eval p|
|---|---|---:|---:|---:|---:|
|P1|>=50|29000|76|0.485500|0|
|P1|>=250|32500|252|0.483750|0|
|P1|>=508|44500|508|0.477750|0|
|P1|512|53500|512|0.473250|0|
|P2|>=50 AND >=250|301500|265|0.349250|0|
|P2|>=508|422000|508|0.289000|0|
|P2|512|447000|512|0.276500|0|
|P3|>=50|844000|51|0.078000|0|
|P3|>=250|948000|250|0.026000|0|
|P3|>=508 /512|censored at1000000|final301|final0.000000|0|
|P4|>=50 /250 /508 /512|censored at1000000|final20, max22|final0.000000|0|

P2 jumps25->265 between301000 and301500, so first-50 and first-250
coincide; this is not a transcription error. Its first-perfect milestone
arrives with27.65% carry probability still present. P3 ignites much later,
with7.8% carry, and first reaches250 with only2.6% carry remaining.
This is acquisition under a diminishing crutch followed by endpoint survival
without it; it does not show that P2/P3 would have ignited from scratch at p=0.

|Update|Training p|P1|P2|P3|P4|
|---:|---:|---:|---:|---:|---:|
|160000 (restart checkpoint)|0.420000|512|21|1|0|
|160500 (new branch)|0.419750|512|19|0|0|
|301500|0.349250|512|265|10|2|
|447000|0.276500|512|512|18|8|
|750000|0.125000|512|512|29|18|
|844000|0.078000|512|512|51|20|
|900000|0.050000|512|512|162|20|
|948000|0.026000|512|512|250|20|
|975000|0.012500|512|512|268|20|
|1000000 (fixed final)|0.000000|512|512|301|20|

## Retention: exact dips, not an assumed monotone wean

This is registered-(b)'s decay/retention question (the task called it '(c)';
see corrections). Define the descriptive **late-anneal window as750K–1M**,
p<=0.125; this is a reporting window, not a new registered success criterion.

|Position|Minimum after first512 (P3: first50)|Late-window minimum|Late maximum drawdown from a preceding peak|Final|
|---|---|---|---|---:|
|P1|491 at205000 (21 below512)|512, no dips|0|512|
|P2|490 at517000 (22 below512)|509 at812500|512->509, 3 counts|512|
|P3|45 at846500 after first50=51|22 at767000 (before material ignition)|145 at886500 ->122 at887500, 23 counts|301|

P1 is not perfectly retained immediately after acquisition: even after447K
it drops to506 at475500; in500K–1M its minimum is509 at587500.
Its last nonperfect observation is511 at708500, then **512 at every validation
from709000 through1M**. P2's largest post-perfect dip is exactly
**512 at516500 ->490 at517000**, not permanent loss. Its last nonperfect
observation is511 at871500, then **512 throughout872000–1M**.

All late-window P2 observations below512 are listed here (17 observations,
16 downward transitions; the759000/759500 dip persists for two evaluations).
All unlisted late P2 observations are512. P1 has no late-window dip.

|Update|P2 /512|
|---:|---:|
|755500|511|
|756500|511|
|759000|511|
|759500|511|
|770000|511|
|776000|511|
|790500|510|
|812500|509|
|814500|511|
|825000|511|
|829000|511|
|833000|511|
|844500|510|
|845500|511|
|865500|511|
|869500|511|
|871500|511|

P3 is noisy while consolidating: **all below50 observations after first-50**
are844500:47,845500:46,846000:46,846500:45,847500:49,850500:48.
It stays>=50 from851000 onward. **All below250 observations after first-250**
are948500:236,950000:243,951500:246,952000:247,952500:245,953000:244,
953500:247,954000:248,954500:246,955000:241,956500:245,957000:242,
957500:248,958500:244,959000:244,960500:249,962000:243,969500:248.
It stays>=250 from970000 onward. The first250 crossing immediately dips
**250 at948000 ->236 at948500**, a14-count loss, p0.026000->0.025750.
The largest single-evaluation late P3 drop is **211 at922500 ->191 at923000**
(20 counts); the largest peak-to-trough late drawdown is the23-count one above.
There are217 downward P3 transitions in750K–1M and48 in948K–1M, not a
smooth monotone curve. These exact threshold recrossings and extrema summarize
the full hash-bound500-step history; they do not treat every small fluctuation
as crutch dependence. At900K–1M, P1/P2 remain512 throughout while P3's minimum
is151 at900500 and its final/max is301. No terminal retention destruction is
observed, but one zero-crutch endpoint is not a sustained p=0 training phase.

## Infrastructure history and authoritative branch

The original run died from root **ENOSPC at update160,668**:160667 is the
last full training row and160668 is a truncated partial row. This was an
infrastructure death, not a scientific failure or a zero score.
The retained checkpoint is160000, SHA256
`d55ce04d3b0a816ed8148bd1358e9ebc4e4718b85b615e32604601c59be83592`.
An exact-replay attempt failed closed at160022:
`loss 2.1394128799438477 != 2.1394338607788086`.
The failed [divergence receipt](infra_resume/e97_carrywean_seed0/interrupted.json)
is retained, SHA256
`7de99e2661e0a9cfecd2094dc115d1e587ec8aca856ceb64a5c2ad3195699c9f`.
No equality tolerance was weakened. CUDA nondeterminism is plausible, not an
independently established cause or proof that the trajectories coincide.

A separately approved [checkpoint-restart amendment](infra_restart/AMENDMENT-2026-10-21-restart-e97_carrywean_seed0.md)
and [supersession receipt](infra_restart/registration-supersession-e97_carrywean_seed0.json)
restore model, AdamW moments and NumPy/torch/CUDA RNG from160000.
Historical updates160001–160667 and partial160668 are **superseded**, NOT
replayed exactly or restored at death. Every fresh training row records this
branch boundary and `replay_verified=false`. Authoritative history is original
validations through160000 plus new160500–1M; the original160500 P2=18 is
superseded by new P2=19. Training is160000 original +840000 restarted updates.
The original global carry/LR schedules and1M ceiling are unchanged.
No uninterrupted-trajectory counterfactual is established by the terminal success.

## Chapter context and honest scope

**This is the second independently designed staging fix demonstrated at T64**:
X16E curriculum reaches **four-perfect** [512,512,512,512,10] after500K teach
+500K extend; X17 carry-wean reaches **two-perfect + P3=58.79%** after1M
updates. Here 'independent' means a distinct intervention, not independent seeds,
replicated evidence or a statistically controlled comparison. X16E bundles a
horizon change and fresh extension cosine; X17 bundles carry-slot format and
supervised examples under one global wean. Neither isolates a single mechanism.

The same-seed cold-start contrast is stark: X12 AdamW T64 seed0 never reaches50
(max any position3), and X15G Muon@3x T64 seed0 never reaches50 (P1 max19,
P2+ max4), through their1M ceilings. Cold failure is the observed outcome of
these prescribed seeded trajectories, **not a universal deterministic law**:
X13 AdamW seed1 also fails, but seed2 escapes to two-perfect + P3=309 at1M.
X17 is not superior to that lottery escape on these terminal counts.
SF has no cold-T64 receipt; its failures at other horizons are not an additional
T64 optimizer comparison. These are historical contrasts, not a matched
randomized causal trial or proof that staging is necessary/all-seed reliable.

Sources: [X16E terminal growth](../results_invertible_stage16_ignition_8m/curriculum_extension/GROWTH_REPORT.md),
[X15G terminal addendum](../results_invertible_stage15_probes_8m/muon_restart/X15G-growth-report-addendum.md),
[X12](../results_invertible_stage12_separation_8m/REGISTRATION.md),
[X13](../results_invertible_stage13_zerocheck_8m/GROWTH_REPORT.md), and their
archived validation histories. X16E's documented teaching/restart caveats remain.

**Scope: single seed, one wean design, one completed checkpoint-restarted run.**
P4 is **budding and censored**, final20, maximum22, never>=50. P3 first508/512
and P4+ material crossings remain right-censored at1M. There is **no endpoint-growth
claim** beyond the observed material set {P1,P2,P3}, no fourth learned position,
no deeper eventual-acquisition prediction, and no proof of unbounded composition.
Zero-crutch survival here is one endpoint, not a long p=0 retention experiment.
Standalone lambda01 expressivity only: no Frontier/Slurm submission, resilient
DiLoCo behavior change, or scaling claim. Resilient DiLoCo requirement IDs are
not applicable to this report-only change. No new training was run.

## Corrections and evidence

**Concession #13:** the task mislabeled retention as prediction '(c)'. The
registration governs: **(b) is decay/crutch-dependence; (c) is P1 timing**.
Both are evaluated separately above; no prediction or registration was rewritten.
The final '[512,512,301,20,0...]' shorthand is not literally all zeros after P5:
P9=2 and P25/P36/P60/P62=1; every other P5–P64 final count is0.

The primary artifacts are under `infra_restart/e97_carrywean_seed0/`.
`result.json` says `status=complete`, `total_steps=1000000`,
`stop_reason=safety_ceiling`, and agrees exactly with the final validation.
The exit0 [worker terminal](infra_restart/receipts/e97_carrywean_seed0/terminal.json)
and [DONE marker](infra_restart/markers/e97_carrywean_seed0.DONE) bind that result.
The final checkpoint SHA256 is
`4eacfcbabfd084d8e2ccf2ccdb1442987cb7ce4104a1bee3791c41a38d7809bc`.

|Artifact (relative to this report)|SHA256|
|---|---|
|infra_restart/e97_carrywean_seed0/result.json|91f9aa3995f8c727b7e10ce8911c8da02c32812756f66e472fcd9d3317d3b5e6|
|infra_restart/e97_carrywean_seed0/validation.json|2cfaf5d1cf0292e93b5718fdf85c23ff45376afe1ff11544b3760f174ff09e0f|
|infra_restart/e97_carrywean_seed0/training.jsonl|c1668109a405993ba50b8a377331bb5566fab8e85a898a5e7a1024f05be6a305|
|e97_carrywean_seed0/validation.json (original, includes superseded160500)|3ad2da03dad4c1a5895c59c330a5de62d96ad7083eedfdbf1d4927f19c864d28|

Fresh read-only checks on lambda01 using `.venv/bin/python` verified all1M
consecutive authoritative training rows against the registered carry formula,
finite loss/grad_norm/LR, restart-row branch provenance, all2001 validation
rows/cadence and neutral eval p, inherited prefix equality through160000,
final/result/terminal/DONE equality, and retained source/final checkpoint hashes.
The crossing p values above agree exactly between training and validation logs;
for example raw float values are0.27649999999999997 at447K,
0.07800000000000001 at844K,0.026000000000000023 at948K, and0.0 at1M.
This is receipt/byte verification, NOT a fresh model replay or a full launch-source
admission audit. Terminal receipts remain local current-checkout artifacts under
the standing report-only commit practice; an independent reviewer must bind them.

Validation command:
`.venv/bin/python -m pytest -q tests/test_invertible_stage17_carrywean_8m.py tests/test_invertible_stage16_17_restart.py`
— **25 passed**. No source/tests/registration/training artifacts were changed.

## Full64-position terminal receipt

Counts are correct/512. Maxima are per-position, not a jointly attained prefix.
Times are first observed crossings in the authoritative branch; censored means
not observed through1M, not a prediction about future training.

|Position|Final|Max|First50 update / carry_p|First250 update / carry_p|First508 update / carry_p|First512 update / carry_p|
|---|---:|---:|---|---|---|---|
|P1|512|512|29000 / 0.485500|32500 / 0.483750|44500 / 0.477750|53500 / 0.473250|
|P2|512|512|301500 / 0.349250|301500 / 0.349250|422000 / 0.289000|447000 / 0.276500|
|P3|301|301|844000 / 0.078000|948000 / 0.026000|censored|censored|
|P4|20|22|censored|censored|censored|censored|
|P5|0|3|censored|censored|censored|censored|
|P6|0|3|censored|censored|censored|censored|
|P7|0|3|censored|censored|censored|censored|
|P8|0|2|censored|censored|censored|censored|
|P9|2|2|censored|censored|censored|censored|
|P10|0|2|censored|censored|censored|censored|
|P11|0|3|censored|censored|censored|censored|
|P12|0|2|censored|censored|censored|censored|
|P13|0|2|censored|censored|censored|censored|
|P14|0|2|censored|censored|censored|censored|
|P15|0|3|censored|censored|censored|censored|
|P16|0|2|censored|censored|censored|censored|
|P17|0|2|censored|censored|censored|censored|
|P18|0|3|censored|censored|censored|censored|
|P19|0|3|censored|censored|censored|censored|
|P20|0|3|censored|censored|censored|censored|
|P21|0|2|censored|censored|censored|censored|
|P22|0|2|censored|censored|censored|censored|
|P23|0|3|censored|censored|censored|censored|
|P24|0|2|censored|censored|censored|censored|
|P25|1|3|censored|censored|censored|censored|
|P26|0|2|censored|censored|censored|censored|
|P27|0|3|censored|censored|censored|censored|
|P28|0|2|censored|censored|censored|censored|
|P29|0|3|censored|censored|censored|censored|
|P30|0|3|censored|censored|censored|censored|
|P31|0|3|censored|censored|censored|censored|
|P32|0|3|censored|censored|censored|censored|
|P33|0|3|censored|censored|censored|censored|
|P34|0|2|censored|censored|censored|censored|
|P35|0|3|censored|censored|censored|censored|
|P36|1|2|censored|censored|censored|censored|
|P37|0|2|censored|censored|censored|censored|
|P38|0|2|censored|censored|censored|censored|
|P39|0|3|censored|censored|censored|censored|
|P40|0|3|censored|censored|censored|censored|
|P41|0|3|censored|censored|censored|censored|
|P42|0|2|censored|censored|censored|censored|
|P43|0|2|censored|censored|censored|censored|
|P44|0|4|censored|censored|censored|censored|
|P45|0|3|censored|censored|censored|censored|
|P46|0|3|censored|censored|censored|censored|
|P47|0|3|censored|censored|censored|censored|
|P48|0|3|censored|censored|censored|censored|
|P49|0|3|censored|censored|censored|censored|
|P50|0|2|censored|censored|censored|censored|
|P51|0|3|censored|censored|censored|censored|
|P52|0|2|censored|censored|censored|censored|
|P53|0|2|censored|censored|censored|censored|
|P54|0|2|censored|censored|censored|censored|
|P55|0|2|censored|censored|censored|censored|
|P56|0|3|censored|censored|censored|censored|
|P57|0|4|censored|censored|censored|censored|
|P58|0|2|censored|censored|censored|censored|
|P59|0|3|censored|censored|censored|censored|
|P60|1|3|censored|censored|censored|censored|
|P61|0|3|censored|censored|censored|censored|
|P62|1|3|censored|censored|censored|censored|
|P63|0|3|censored|censored|censored|censored|
|P64|0|2|censored|censored|censored|censored|
