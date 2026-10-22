# X15G terminal growth-report addendum — Muon@3x NO-ESCAPE at 1M

## Registered question and terminal answer

Does hybrid-Muon@3x, the T10-validated operating point, change cold-T64
ignition? **NO in this seed0 run: NO REGISTERED ESCAPE through the completed
1,000,000-update ceiling.** No position ever reached50/512; fixed-final
material positions are empty, and the perfect prefix is0. P1 finishes6/512,
with maximum19 first at41K (tied at84K), both BEFORE the interruption. It
never climbed above that flicker. P2–P64 maximum anywhere is4/512: noise-scale
counts, **zero material composition**, not literally zero correct guesses.
P1 is extraction/non-composition, not evidence of iterative composition.

Fixed-final validation CE is **8.159796715**, versus exact marginal
ln(3721)=8.221747728: **-0.061951014**, about0.062 BELOW marginal. This is a
flat below-marginal P1-extraction signature, not an exact-marginal equality
or proof of no learning. At final P1 x-coordinate is512/512 but joint-state
accuracy is6/512; its CE4.115609549 is near ln(61)=4.110873864. P2+ mean CE
is8.223990344. P1's reduced CE explains the aggregate foothold without a
material joint-state crossing or composition. No post-score equivalence
tolerance or chance hypothesis test is introduced.

## Registered criteria, verbatim, and their application

The immutable [REGISTRATION.md](../REGISTRATION.md) remains the authority;
this addendum is the outcome document, not a revised registration.

> (a) If any position reaches>=50/512, Muon@3x escapes the marginal at T64 where
> historical AdamW+cosine could not: supports optimizer-dependent learning failure
> at these operating points; direction-normalized updates can climb the1/64 foothold.
> Transient first crossing is descriptive; fixed-final retention reported separately.
> (b) If Muon also sits at exact-marginal through1M, failure is optimizer-robust
> across these AdamW/Muon operating points, strengthening the horizon/foothold-share
> account (both optimizers escaped atT16). A one-seed conditional result, not a
> universal optimizer ranking or proof of impossibility.

**(a) Not met:** maximum any position19<50; no transient escape and no
fixed-final retained escape. The historical wording is quoted, not silently
updated: X13 seed2 subsequently supplied a cold AdamW lottery escape.
**(b) Its literal exact-marginal premise is not established:** CE is below
marginal, with real P1 coordinate extraction. The terminal finding is instead
no material ignition/composition at this operating point, consistent with the
cold AdamW failures. The escape bar alone does not establish exact marginality,
zero learning, or universal optimizer-robust impossibility. The all64 count/CE
signature and flat trajectory, not an invented numerical tolerance, bound the
interpretation.

All first-50 and first-512 times below remain **right-censored at1M** (no
observed event in the registered window), not time0, infinity, or a prediction
about later training. The completed no-escape verdict is no longer an interim
infrastructure-censored observation. Validation cadence is500 updates; no
unsampled transient is claimed.

## Trajectory: the plateau spans the restart boundary

Counts are correct/512. Full2001-row, all64-position count/CE history is in
`muon3x_seed0/validation.json` and `marginal_signature.json`; selected rows:

|Update|Branch/readout|P1|Max P2+|Validation CE|CE - ln3721|
|---:|---|---:|---:|---:|---:|
|41000|Original; first P1 maximum|19|1|8.159782767|-0.061964961|
|84000|Original; tied P1 maximum|19|2|8.159656286|-0.062091442|
|100000|Original|7|1|8.159446299|-0.062301429|
|200000|Original|9|1|8.159668326|-0.062079402|
|300000|Original|10|1|8.159396470|-0.062351259|
|400000|Original; start of last600K|11|1|8.158791006|-0.062956723|
|430000|Original; before interruption|11|1|8.158653319|-0.063094409|
|433000|Original checkpoint / restart boundary|8|2|8.158940017|-0.062807711|
|433500|Restart; first new validation|10|1|8.158906043|-0.062841686|
|434000|Restart|7|1|8.159059644|-0.062688085|
|450000|Restart|12|2|8.159177303|-0.062570425|
|500000|Restart|10|2|8.158822000|-0.062925728|
|600000|Restart|12|2|8.158697784|-0.063049944|
|700000|Restart|10|1|8.158322513|-0.063425215|
|800000|Restart|10|1|8.157908857|-0.063838871|
|900000|Restart|8|1|8.159240007|-0.062507721|
|1000000|Restart; fixed final|6|1|8.159796715|-0.061951014|

Over400K–1M, validation CE ranges8.157521725–8.165422976; “flat600K”
describes this bounded plateau, not identical losses or a statistical test.
P1 is already down to8 at433K after the early19 peak, then10/7 immediately
after the boundary and6 at final. **The restart did not truncate an observed
climb:** the plateau exists on both sides. This does not prove what the lost
uninterrupted counterfactual would have done.

## Infrastructure history, provenance, and corrections

1. Original run died with ENOSPC at433,296; last committed checkpoint433,000.
   This was infrastructure interruption, not an optimizer failure verdict.
2. [Amendment-1](../muon_resume/AMENDMENT-2026-10-21.md) attempted exact replay.
   Checkpoint restoration passed and updates433001–433043 matched diagnostics;
   update433044 differed in `muon_update_rms` by about5.23e-11
   (0.11409317095890362 versus0.11409317090657782). It failed closed, without
   tolerance, before fresh433297+ updates. This is consistent with CUDA
   nondeterminism, not an independently isolated causal diagnosis. Its
   `interrupted.json` and FAIL/terminal evidence remain INFRA-INTERRUPTED.
3. Separately authorized [amendment-2](AMENDMENT-2-2026-10-21.md) restarted from
   checkpoint433000, exactly restoring model, complete Muon/auxiliary AdamW
   optimizer and RNG state as receipted in
   `muon3x_seed0/checkpoint_restored.json`. That receipt explicitly records
   `at_death_restoration_claimed=false`. Historical433001–433296 are superseded
   by re-execution, NOT bitwise replayed or stitched into the terminal branch.
   Original validations through433000 plus new433500–1M form the history;
   567,000 fresh restart updates retain the original1M cosine and ceiling.
4. [Amendment-3](AMENDMENT-3-2026-10-21.md) grants an evidence concession for
   unreceipted pruning by this already-running process. **Deletion-time hashes,
   timestamps and deleted-file identity are unproved.** Index hashes are
   write-time receipts only. The prospective fix did not retrofit the running
   process; this terminal report does not declare the historical omission
   compliant or fabricate pruning receipts.

**Concession #9 retained:** task text's500K ceiling was wrong; the registered
1M ceiling/cosine governs. **Concession #12 (terminal task correction):** there
is NO cold-T64 SF receipt. SF evidence is at its tested T16/T128 horizons,
not T64; the task's synthesis shorthand was the error, not the registrations.
No original registration, amendment, failed-replay receipt or training artifact
is rewritten by this report.

## Chapter synthesis — staging works; changing optimizer did not rescue cold T64

|Evidence|Observed outcome|Scope|
|---|---|---|
|Cold T64 AdamW, X12/X13|2/3 seeds dead through1M; seed2 lottery escape reaches two-perfect + P3=309 at1M|Cold ignition can happen; not a universal zero or impossibility result|
|Cold T64 hybrid-Muon@3x, this run|No escape at1M; P1 max19, P2+ max4, final P1=6|One seed, one T10-validated operating point, interrupted/checkpoint-restarted branch|
|SF-class at tested horizons|Cold long-horizon failure at T16 (X7/X9b flat) and T128 (X3 replicated stagnation)|**Untested cold at T64**; not a matched T64 optimizer comparison or a common1M endpoint|
|Staged T64, X16E|Fixed-final P1–P5=[512,512,512,512,10] after500K teach +500K extend; T16 retention also four-perfect|Prescribed deterministic seeded procedure, NOT replicated all-seed reliability|

Sources: [X12 registration](../../results_invertible_stage12_separation_8m/REGISTRATION.md)
and its seed0 validation; [X13 growth receipt](../../results_invertible_stage13_zerocheck_8m/GROWTH_REPORT.md);
SF `results_invertible_stage7_deepfirst_8m` / `results_invertible_stage9b_sf_anneal_8m`
config/result receipts and `results_invertible_stage3_8m/results_summary.json`;
[X16E terminal report](../../results_invertible_stage16_ignition_8m/curriculum_extension/GROWTH_REPORT.md).

**Cold T64 fails under both tested optimizer classes in the dead trajectories;
staging is the property that works in the demonstrated recipe, not simply the
optimizer switch.** T64 optimizer-robustness here rests on AdamW+Muon only;
SF failure is historical context at other tested horizons. The AdamW lottery
escape rules out “cold T64 always fails.” X16E's four-perfect endpoint is a
successful staged route, not proof that every curriculum works, staging is
necessary, or its mechanism is isolated. X16E bundles staging/horizon and a
fresh extension cosine, and its teaching run has a documented restart caveat.
These are historical endpoint contrasts, not a new matched randomized ablation
or universal optimizer ranking. P5 has not crossed50; no deeper growth is
implied. Infrastructure-censored attempts are never counted as optimizer deaths.

## Terminal evidence audit

[receipts/terminal_audit.json](receipts/terminal_audit.json) records the fresh
read-only audit on lambda01: all186 source bindings match admission785dd40c,
all388 dependency hashes and five original artifact hashes match, and all three
frozen-pool array hashes match. Current runner/tests differ from admission only
as disclosed in amendment-3; admission blobs, not changed source, attest launch.
The original checkpoint and retained aggregate-best/final checkpoint bytes hash
correctly. This is hash/receipt verification, not a new full-state replay.

The audited branch has433,000 original +567,000 fresh consecutive finite
updates, each LR exactly equal to the original cosine; every restart row has
branch provenance. All2001 validations are consecutive at500-step cadence,
with inherited history matching the original and final matching the bound
result. The complete all64 signature matches the validation history. Terminal
result SHA256 is `d2667be22225ee05298b4ede10506cadd302ef4d55be00727162478f432db807`;
final checkpoint SHA256 is
`dfe4036faf9118b6ef6209c8215a887d5134db9b35fd9621d54488dcd805ab5c`.
Worker exit0/DONE binds the1M safety-ceiling result; native watcher log records
both probes DONE at2026-10-22T06:26:02Z (local watcher06:25:23Z).
The amendment-3 pruning gap and explicit denial of at-death restoration remain
limitations, not erased by successful terminal auditing. Independent review is
still required; this report does not award itself acceptance.

Standalone expressivity only: Frontier queue evidence and resilient-DiLoCo
requirement IDs are not applicable. No training, relaunch, tuning or extension
was performed for this report.

## All64 fixed-final positions, maxima, and censored crossing times

|Position|Latest /512|Max /512|First>=50|First512|Latest CE|
|---|---|---|---|---|---|
|P1|6|19|censored@1000000|censored@1000000|4.115609549|
|P2|0|3|censored@1000000|censored@1000000|8.226197275|
|P3|0|3|censored@1000000|censored@1000000|8.235415356|
|P4|0|3|censored@1000000|censored@1000000|8.224270702|
|P5|0|3|censored@1000000|censored@1000000|8.226385300|
|P6|1|3|censored@1000000|censored@1000000|8.218886292|
|P7|0|2|censored@1000000|censored@1000000|8.232497610|
|P8|0|2|censored@1000000|censored@1000000|8.227475386|
|P9|0|2|censored@1000000|censored@1000000|8.218053597|
|P10|0|2|censored@1000000|censored@1000000|8.220915379|
|P11|0|3|censored@1000000|censored@1000000|8.223566730|
|P12|0|3|censored@1000000|censored@1000000|8.222754632|
|P13|0|3|censored@1000000|censored@1000000|8.221206194|
|P14|0|3|censored@1000000|censored@1000000|8.224497660|
|P15|0|2|censored@1000000|censored@1000000|8.230166085|
|P16|0|2|censored@1000000|censored@1000000|8.228516583|
|P17|1|3|censored@1000000|censored@1000000|8.224874796|
|P18|1|2|censored@1000000|censored@1000000|8.223245362|
|P19|0|3|censored@1000000|censored@1000000|8.222932261|
|P20|0|2|censored@1000000|censored@1000000|8.224183451|
|P21|0|2|censored@1000000|censored@1000000|8.224711704|
|P22|0|3|censored@1000000|censored@1000000|8.224297401|
|P23|0|3|censored@1000000|censored@1000000|8.225350453|
|P24|0|2|censored@1000000|censored@1000000|8.222816263|
|P25|0|3|censored@1000000|censored@1000000|8.219083893|
|P26|0|4|censored@1000000|censored@1000000|8.222213009|
|P27|1|3|censored@1000000|censored@1000000|8.225344245|
|P28|0|2|censored@1000000|censored@1000000|8.220803260|
|P29|0|3|censored@1000000|censored@1000000|8.221986402|
|P30|0|3|censored@1000000|censored@1000000|8.226071855|
|P31|0|3|censored@1000000|censored@1000000|8.224908845|
|P32|0|2|censored@1000000|censored@1000000|8.223211644|
|P33|0|2|censored@1000000|censored@1000000|8.222931993|
|P34|0|3|censored@1000000|censored@1000000|8.228146844|
|P35|1|3|censored@1000000|censored@1000000|8.223110659|
|P36|0|2|censored@1000000|censored@1000000|8.222624873|
|P37|0|3|censored@1000000|censored@1000000|8.228045680|
|P38|0|2|censored@1000000|censored@1000000|8.221895150|
|P39|0|3|censored@1000000|censored@1000000|8.224508012|
|P40|0|2|censored@1000000|censored@1000000|8.225133873|
|P41|1|3|censored@1000000|censored@1000000|8.224572346|
|P42|0|4|censored@1000000|censored@1000000|8.225143553|
|P43|0|2|censored@1000000|censored@1000000|8.221455258|
|P44|0|2|censored@1000000|censored@1000000|8.220383653|
|P45|0|3|censored@1000000|censored@1000000|8.223780090|
|P46|0|3|censored@1000000|censored@1000000|8.222064297|
|P47|0|3|censored@1000000|censored@1000000|8.221718490|
|P48|0|3|censored@1000000|censored@1000000|8.222812722|
|P49|0|3|censored@1000000|censored@1000000|8.227032091|
|P50|1|3|censored@1000000|censored@1000000|8.221154855|
|P51|0|2|censored@1000000|censored@1000000|8.223924192|
|P52|0|2|censored@1000000|censored@1000000|8.222247178|
|P53|0|2|censored@1000000|censored@1000000|8.223394939|
|P54|0|2|censored@1000000|censored@1000000|8.220459461|
|P55|0|3|censored@1000000|censored@1000000|8.224677855|
|P56|1|3|censored@1000000|censored@1000000|8.223831141|
|P57|0|3|censored@1000000|censored@1000000|8.226087561|
|P58|0|3|censored@1000000|censored@1000000|8.223184487|
|P59|1|3|censored@1000000|censored@1000000|8.224917964|
|P60|1|3|censored@1000000|censored@1000000|8.221447972|
|P61|0|3|censored@1000000|censored@1000000|8.221791842|
|P62|0|3|censored@1000000|censored@1000000|8.227693358|
|P63|0|3|censored@1000000|censored@1000000|8.223311988|
|P64|0|2|censored@1000000|censored@1000000|8.221067657|

## Training loss trajectory (full per-update records in bound streams)
Original rows after433000 are superseded, not stitched into this branch.
/home/erikg/emender/experiments/expressivity_tasks/results_invertible_stage15_probes_8m/muon3x_seed0/training.jsonl: update1 loss=250.876220703; update433000 loss=8.158800125.
/home/erikg/emender/experiments/expressivity_tasks/results_invertible_stage15_probes_8m/muon_restart/muon3x_seed0/training.jsonl: update433001 loss=8.159132004; update1000000 loss=8.160743713.
