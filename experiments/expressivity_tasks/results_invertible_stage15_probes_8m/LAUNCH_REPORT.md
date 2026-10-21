# X15 — two probes: launch and zero-shot report

## Prelaunch chronology / approved correction

Original registration **2026-10-21 14:36:56 UTC**, SHA256
`2ccea98a6026855603ad5ea74b54e36e530619714e7d7d22a81d6a546bdfd2c2`;
source/preregistration commit **5d51341a** pushed before either launch.
171 project/source and388 dependency files verified against committed/worktree
bytes; no stage12 runner in the closure. Cards6/7 exclusively leased; cards0–5
and their jobs untouched. No WorksGood service/dispatcher was used.

The original GDN2 attempt failed closed **before T64 evaluation**: hash-pinned
aggregate-best482500 reproduced all16 historical T16 counts exactly, then the
extra hardcoded assertion copied the task's incorrect P5=318 expectation.
Supervisor acknowledged the task had conflated aggregate-best P5=333 with final
P5=318 and approved exactly one attended correction/relaunch, same champion/hash.
Original failure evidence and registration/source remain immutable; in particular
the running Muon closure was NOT patched. GDN2-only script, own registration and
[amendment](gdn2_amendment/AMENDMENT.md) committed/pushed as **6c073c0e** before
corrected T64 evaluation. Amendment registration **14:41:48 UTC**, SHA256
`f28959ecc25d7319c162000905cb73af73fd258412dbbb13f620ec0faa9ffff4`.
Predictions(a)/(b)/(c) unchanged, champion P5 baseline corrected333, not318.

## Probe1: GDN2 zero-shot complete on physical card6

Corrected worker1220688; trainer identity and successful terminal/DONE/result
hash binding retained under `gdn2_amendment/receipts/gdn2_zeroshot/`.
X11 aggregate-best482500 SHA256
`49a0f96fe7890ca9bb8a860698641ad8407dca412a3b38a01d3ede996c011cd0`.
Strict stock-X11 model load, trusted stage10/X11 frozen pool/forward/padder,
stock bf16 FLA GDN2 (no fp32 adapter). **Zero training updates**, no optimizer;
checkpoint hash and all model state unchanged. Historical T16 counts reproduced
exactly before T64. Full64 per-position counts/CE in `gdn2_amendment/gdn2_zeroshot/result.json`
and growth report; requested P1–P16 comparison:

|Position|Champion T16 /512|Zero-shot T64 /512|
|---|---:|---:|
|P1|512|512|
|P2|512|512|
|P3|512|512|
|P4|512|511|
|P5|333|295|
|P6|3|2|
|P7|0|0|
|P8|0|0|
|P9|0|0|
|P10|0|0|
|P11|0|0|
|P12|0|0|
|P13|0|0|
|P14|0|0|
|P15|0|0|
|P16|0|0|

Prediction**(a) supported in its registered approximately512 sense**: trained
P1–P4 retain512,512,512,511 on longer contexts, not a collapse; both architectures
show runtime generalization beyond training horizon. Not an exact-four-perfect
claim: P4 loses one example. Prediction(b)'s collapse/asymmetry not observed.
Prediction**(c)**: P5 declines333→295 rather than consolidating zero-shot; P6=2.
This is runtime evaluation, NOT evidence that GDN2 learned T64 composition from
scratch, and no global long-horizon accuracy claim. Full P17–P64 remain chance-sized.
The generated watcher's stricter exact512 branch conservatively prints raw counts;
the approximately512 interpretation above follows the actual registration wording.

## Probe2: Muon@3x scratch T64 running on physical card7

Worker **1215981**, trainer **1216577**, starttime identities verified; physical
UUID **GPU-8f63894f-5b64-755c-060f-7022c603e7d2** verified by nvidia-smi and lease
ownership/heartbeat. Scratch seed0, E97 legacy-bf169,900,800 parameters; uniform
P1–P64 CE, batch32, clip1, exact unchanged X10 HybridMuon/routing/RMS matching,
all-parameter decay.01. Warmup5K→3e-3, **cosine stretched to3e-5 at1M**, explicitly
approved before registration to match historical T64 full-ceiling schedule shape.
No stage12 code path. No metric stop, tuning, restart or automatic retry.

Immutable launch audit at **14:43:30 UTC**: **6,504 consecutive finite updates**,
LR exactly matches registered schedule at every update; validation through**6,500**,
all64 positions/aggregate reconciliation and frozen T64 array/archive hashes pass.
Entry checkpoint has empty Muon/auxiliary AdamW state, bound route list and SHA256
`cfa24a1fa9110d26f64843dc081ca54facd2f25e9f9ed3e0e7979b9dcdb5b090`.
Two retained checkpoints at audit, bounded entry+best+latest/final policy unchanged.
Wall310887s /8GiB memory guards; infrastructure failures are censored.

At6,500 validation CE **8.175311446**, versus ln3721 **8.221747728**, difference
**−0.046436282**; max position **9/512**, no material>=50 crossing yet. This is
an **early, censored snapshot**, not exact-marginal or optimizer-robust-failure
adjudication. Initial validation CE251.382983;5K CE8.202961. Per-update training
loss alongside full500-step validation loss/all64 count/CE series in live logs
and generated `marginal_signature.json`. Near-marginal CE is NOT proof of zero
learning. Predictions(a)/(b) remain pending to fixed-final1M, with transient
crossings and final retention separated. No X12 causal interpretation imported.

## Checks / reproducible launch evidence

- CPU X15+adjacent X10/X11: **37 passed,6 skipped**.
- Leased card7 X15 GPU suite: **11 passed**, including actual all64 eval.
- Disposable seed99 peak-LR25-update T64 probe: finite; reload-next-update max
  difference**0.0**, peak687458304 bytes; no confirmatory frozen pool used.
- Approved amendment plus original X15 CPU tests: **12 passed,1 skipped**;
  tests cover corrected333 strict-load/control, zero updates/state, amended watcher
  terminal binding and separation from preserved original failure.
- Postlaunch X15/amendment/adjacent X10/X11 CPU regressions: **39 passed,6 skipped**.
- Both registrations reverified after amendment (171/186 project files and388
  dependencies respectively); original sources unchanged. All10 immutable launch
  snapshot manifest hashes reverified.
- Launch audit command (one-shot, refuses overwrite):
  `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.expressivity_tasks.results_invertible_stage15_probes_8m.receipts.verify_launch`.
  `receipts/launch_snapshot/` and SHA256 `receipts/launch_manifest.json` freeze
  config/data/validation/index/lease/UUID/GDN2 result and verification summary.
- Watcher `--once --report`: expected exit**2**, DONE GDN2 / RUNNING Muon.
- Source diff whitespace checks clean. Raw prelaunch nvidia-smi receipt contains
  two original trailing-whitespace lines; preserved immutable rather than edited
  after hashing (`core.whitespace=-blank-at-eol` used for prereg diff check).
- Unrelated existing dirty files left untouched; no files intentionally staged.
  Standalone expressivity only: Frontier queue/DiLoCo requirement IDs inapplicable.

Launch commands, from `/home/erikg/emender`:
```bash
.venv/bin/python -m scripts.invertible_stage15_control launch --name gdn2_zeroshot
.venv/bin/python -m scripts.invertible_stage15_control launch --name muon3x_seed0
# After approved, separately pushed amendment (one attended retry only):
.venv/bin/python -m scripts.invertible_stage15_gdn2_amendment launch
```

## Native supervisor arming / wake

**Use the amended combined watcher**, not the original which correctly retains
FAIL for the preserved preflight attempt:
```bash
cd /home/erikg/emender
.venv/bin/python -m scripts.watch_invertible_stage15_amended --interval 900
```
Supervisor was notified and explicitly owns native wake arming. Supplying this
command is not a claim that supervisor-side wake has been armed. Exit0 requires
both hash-bound successful results; exit1 failure/dead/reused identity; --once
exit2 pending. It writes combined `GROWTH_REPORT.md` plus full exact-marginal series
on terminal wake. Current snapshot: add `--once --report`. Reboot requires human
rearming; no automatic retry. On wake, audit full1M consecutive finite LR/loss
streams, final checkpoint/terminal/source identities and retention, independently
review predictions, then commit/push the final growth report. Independent reviewer
acceptance remains with the supervisor.
