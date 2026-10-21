# X10b — completed Muon LR-sweep growth report

**3x completes P4; 10x collapses to P1-only composition.** At the fixed **500,000-update** endpoint, the X10 1x hybrid-Muon baseline is `[512,512,512,448]` on P1–P4, the 3x arm is `[512,512,512,512]` with **P5=12/512 budding**, and the 10x arm is `[512,0,0,0]`. **LR-UNDERSCALED is confirmed at 3x; SCALE-TOO-HOT is confirmed at 10x for this learned solution.** The observed operating point is **~3x the AdamW-matched LR at this historical 8M scale**, not a globally optimal LR or general optimizer ranking.

Evidence root **C/**: `experiments/expressivity_tasks/results_invertible_stage10b_muonlr_8m/`. The immutable [registration](experiments/expressivity_tasks/results_invertible_stage10b_muonlr_8m/registration.json), [registered predictions](experiments/expressivity_tasks/results_invertible_stage10b_muonlr_8m/REGISTRATION.md) and unchanged source were committed as **`502329abbf96018616529807bbd78f771091bb56` before launch**; see the [launch report](X10b-launch-report.md). Historical baselines are the registered frozen `C/baselines/x10_{muon,adamw}.json` snapshots, also reported in [X10](X10-growth-report.md). Both higher-LR arms are **DONE**, complete at500K with stop reason **safety_ceiling**, successful terminal/result hash bindings, **500,000 consecutive finite training rows and 1,001 validations each**. Watcher verification returned0. No training, model loading, GPU evaluation, continuation or new seed was performed for this report.

## 1. Three-outcome-column sweep summary

All counts are **joint-state correct out of512**, not coordinate accuracy. Vectors print **all P1–P16**; aggregate denominator8192. Maxima are independent per-column maxima across1,001 validations, **not one checkpoint**. Perfect-prefix depth requires preceding positions simultaneously512/512.

| Readout | X10-muon 1x: 1e-3 peak | muonlr3x: 3e-3 peak | muonlr10x: 1e-2 peak |
|---|---|---|---|
| Fixed-final P1–P16 /512 | `512,512,512,448,0,0,0,0,0,0,0,0,0,0,0,0` | `512,512,512,512,12,0,0,0,0,1,1,0,0,0,0,0` | `512,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0` |
| Max-anywhere P1–P16 /512 | `512,512,512,450,2,2,4,2,2,3,2,2,3,2,3,2` | `512,512,512,512,17,3,2,2,2,2,2,2,2,3,3,2` | `512,2,3,2,2,2,4,3,3,2,2,2,2,3,2,2` |
| Final correct/8192 | 1984 | 2062 | 514 |
| Final CE (nats) | 6.203891933 | 5.913255215 | 7.703034997 |
| Final / maximum simultaneous perfect-prefix depth | 3 /3 | **4 /4** | **1 /1** |
| P3 first512: update / elapsed seconds | **281,500 /16,661.497s** | **375,000 /19,307.735s** | not reached by500K |
| P4 first512: update / elapsed seconds | **not reached; censored at500K** | **377,500 /19,431.050s** | not reached by500K |
| P5 final / maximum | 0 /2 | **12 /17 (budding, not material acquisition)** | 0 /2 |
| Final terminal P16 /512 | 0 | 0 | 0 |
| Final suffix7-restricted P9–P16 /4096 | 0 | 2 | 1 |
| End-to-end wall seconds | 28,071.817 | 25,780.605 | 26,364.038 |
| Summed optimizer seconds (including auxiliary AdamW) | 17,569.074 | 15,968.123 | 16,477.926 |
| Summed training-update seconds | 27,582.441 | 25,313.266 | 25,921.730 |

**One seed per arm, seed0 only**, with the same initialization, procedural sample order and frozen512-example pool. Historical “8M” means **9,900,800 trainable parameters**: scratch E97 T16, dim256/four layers, modulus61/vocab3721, batch32, uniform target CE, clip1, fp32 parameters/bf16 autocast, legacy recurrence and sequential Triton evaluation. No ladder or auxiliary carry/intermediate-state loss. Readout uses **ordinary live parameters** for all Muon arms and AdamW.

Only LR scale changes: **both hybrid groups** follow the same5K linear warmup and cosine shape, with peak/floor **1e-3/1e-5**, **3e-3/3e-5**, or **1e-2/1e-4** at500K. The pinned X10 hybrid rule, semantic36-matrix routing, auxiliary AdamW, momentum0.95/Nesterov, five FP32 Newton–Schulz iterations and approximate update-RMS scaling `0.2*sqrt(max(rows,cols))` are unchanged. All-parameter nominal decay0.01, including explicitly ignored no-weight-decay tags, is retained; scaling LR also scales the recipe's integrated shrinkage. This is a controlled LR-recipe comparison, not isolated attribution to orthogonalization, matrix LR alone or decay alone.

**Marginal/suffix discipline:** uniform joint marginal **1/3721 =0.0268744961%**; uniform CE **ln3721 =8.221747728 nats**. The suffix-only floor concerns observers missing at least two iid drivers, not prefix-aware recurrent models. The chance-sized suffix counts printed above do not establish deep tracking.

## 2. Prediction adjudication and observed pace trade

1. **LR-UNDERSCALED: CONFIRMED at3x.** The preregistered label is misspelled `LR-UNDERSCCALED`; its criterion was materially improved final P4 (e.g. >=508/512), or earlier same-threshold P4 acquisition. At3x, final P4 is **512 versus448 at1x**, and first P4>=50 is **167,500 versus348,500**. First perfect P4 is377,500 and final perfect depth4 is retained. The X10 P4 lag was an **LR-scale artifact for this tested learned solution**, not an intrinsic inability of this Muon rule to compose P4.
2. **SCALE-TOO-HOT: CONFIRMED at10x.** Final P2–P4 are **all zero**; their full-trajectory maxima are **2,3,2/512**. No P2+ position reaches50/512; the full maxima vector above makes this total composition failure beyond P1 explicit. Even P1 first reaches512 only at454,000. This is prefix destruction **relative to the successful1x/3x solutions**, not evidence that10x first learned and then forgot a perfect deeper prefix. The finite complete stream shows a learning failure, **not a numerical crash or infrastructure censoring**.
3. **Registration reconciliation:** original prediction(b), `SCALE-RIGHT`, bundled higher-LR degradation with an inference favoring1e-3-class and a method-intrinsic P4 lag. Its degradation condition holds at10x but **not at3x**. The broader inference is **not supported**: 3x completes P4. `SCALE-TOO-HOT` is the present interpretation of10x, not a retroactive replacement of the frozen registration. The mixed multipliers identify a useful local operating point **around3x**, without forcing both into one binary outcome or claiming a fully tuned optimum.
4. **Pace trade, as observed:** 1x is earlier to **perfect P2/P3** (186,000/281,500) than3x (310,000/375,000); perfect P3 is **93,500 updates later** at3x, but3x completes P4. This is not uniform slower acquisition: 3x reaches **material P2/P3/P4 earlier** and perfect P1 earlier. The completed receipt's exact1x first-P3 value is281,500, not a mid-run approximation. No monotonicity or optimality claim follows from these crossings on the500-update grid.
5. **No wall-time win over AdamW.** Historical paired AdamW reaches P3/P4-perfect at306,500/391,500 and6,890.168/8,605.185s. At3x, P4-perfect is14,000 updates earlier but takes19,431.050s; total25,780.605s remains about**2.45x** AdamW's10,540.194s at500K. The three Muon end-to-end times are observed on their launched cards, not hardware-normalized or equal-tuned comparisons. The original **Muon composes; no wall-time win** verdict stands, with its LR-scale correction.

**P5 censoring:** 3x's final12/512 and maximum17/512 are descriptive **budding**, below the registered50/512 materiality bar, not P5 mastery or depth5. No arm reaches material P5 by500K. Whether P5 later consolidates is **censored, not resolved**: neither an asymptotic plateau nor a promised eventual success. Likewise1x P4 perfection remains censored for that specific run;3x does not retroactively complete it. No capacity bound, cross-seed robustness, production promotion or Stage2 authorization is inferred. The original Stage-1 multi-seed R1/R2/R3 gates remain unadjudicated.

## 3. Registered both-baseline readouts

The two tables below preserve the preregistered **higher-LR arm | X10-muon@1e-3 | X10-adamw@1e-3** outcome columns. First>=512 means first512 exactly (denominator512). Each crossing prints **update / runner elapsed seconds**, not training-only time; crossings need not be retained. “Not reached (censored)” means only no crossing by500K, including the clearly failed10x arm; it is not an infrastructure failure designation. Max-anywhere is never substituted for fixed-final.

### muonlr3x: fixed-final500K

| Position/readout | muonlr3x | X10-muon@1e-3 | X10-adamw@1e-3 |
|---|---:|---:|---:|
| P1 first>=50 update/seconds | 26,500 / 1422.721s | 19,000 / 1646.557s | 24,000 / 689.050s |
| P1 first>=512 update/seconds | 73,000 / 3841.769s | 100,000 / 6707.446s | 50,000 / 1335.961s |
| P1 max-anywhere | 512/512 | 512/512 | 512/512 |
| P1 fixed-final | 512/512 | 512/512 | 512/512 |
| P2 first>=50 update/seconds | 68,000 / 3594.949s | 128,000 / 8257.525s | 187,500 / 4557.050s |
| P2 first>=512 update/seconds | 310,000 / 15992.095s | 186,000 / 11430.645s | 223,000 / 5186.064s |
| P2 max-anywhere | 512/512 | 512/512 | 512/512 |
| P2 fixed-final | 512/512 | 512/512 | 512/512 |
| P3 first>=50 update/seconds | 99,000 / 5199.363s | 189,500 / 11613.969s | 231,500 / 5357.632s |
| P3 first>=512 update/seconds | 375,000 / 19307.735s | 281,500 / 16661.497s | 306,500 / 6890.168s |
| P3 max-anywhere | 512/512 | 512/512 | 512/512 |
| P3 fixed-final | 512/512 | 512/512 | 512/512 |
| P4 first>=50 update/seconds | 167,500 / 8696.618s | 348,500 / 20410.467s | 309,000 / 6933.623s |
| P4 first>=512 update/seconds | 377,500 / 19431.050s | not reached (censored) | 391,500 / 8605.185s |
| P4 max-anywhere | 512/512 | 450/512 | 512/512 |
| P4 fixed-final | 512/512 | 448/512 | 512/512 |
| P5 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P5 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P5 max-anywhere | 17/512 | 2/512 | 15/512 |
| P5 fixed-final | 12/512 | 0/512 | 7/512 |
| P6 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P6 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P6 max-anywhere | 3/512 | 2/512 | 2/512 |
| P6 fixed-final | 0/512 | 0/512 | 0/512 |
| P7 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P7 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P7 max-anywhere | 2/512 | 4/512 | 2/512 |
| P7 fixed-final | 0/512 | 0/512 | 1/512 |
| P8 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P8 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P8 max-anywhere | 2/512 | 2/512 | 3/512 |
| P8 fixed-final | 0/512 | 0/512 | 0/512 |
| P9 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P9 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P9 max-anywhere | 2/512 | 2/512 | 2/512 |
| P9 fixed-final | 0/512 | 0/512 | 0/512 |
| P10 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P10 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P10 max-anywhere | 2/512 | 3/512 | 2/512 |
| P10 fixed-final | 1/512 | 0/512 | 0/512 |
| P11 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P11 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P11 max-anywhere | 2/512 | 2/512 | 2/512 |
| P11 fixed-final | 1/512 | 0/512 | 0/512 |
| P12 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P12 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P12 max-anywhere | 2/512 | 2/512 | 3/512 |
| P12 fixed-final | 0/512 | 0/512 | 0/512 |
| P13 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P13 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P13 max-anywhere | 2/512 | 3/512 | 2/512 |
| P13 fixed-final | 0/512 | 0/512 | 0/512 |
| P14 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P14 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P14 max-anywhere | 3/512 | 2/512 | 3/512 |
| P14 fixed-final | 0/512 | 0/512 | 0/512 |
| P15 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P15 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P15 max-anywhere | 3/512 | 3/512 | 3/512 |
| P15 fixed-final | 0/512 | 0/512 | 0/512 |
| P16 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P16 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P16 max-anywhere | 2/512 | 2/512 | 2/512 |
| P16 fixed-final | 0/512 | 0/512 | 0/512 |

Fixed-final contiguous perfect prefix=4; terminal P16=0/512; suffix7-restricted=2/4096.


### muonlr10x: fixed-final500K

| Position/readout | muonlr10x | X10-muon@1e-3 | X10-adamw@1e-3 |
|---|---:|---:|---:|
| P1 first>=50 update/seconds | 430,500 / 22603.639s | 19,000 / 1646.557s | 24,000 / 689.050s |
| P1 first>=512 update/seconds | 454,000 / 23852.423s | 100,000 / 6707.446s | 50,000 / 1335.961s |
| P1 max-anywhere | 512/512 | 512/512 | 512/512 |
| P1 fixed-final | 512/512 | 512/512 | 512/512 |
| P2 first>=50 update/seconds | not reached (censored) | 128,000 / 8257.525s | 187,500 / 4557.050s |
| P2 first>=512 update/seconds | not reached (censored) | 186,000 / 11430.645s | 223,000 / 5186.064s |
| P2 max-anywhere | 2/512 | 512/512 | 512/512 |
| P2 fixed-final | 0/512 | 512/512 | 512/512 |
| P3 first>=50 update/seconds | not reached (censored) | 189,500 / 11613.969s | 231,500 / 5357.632s |
| P3 first>=512 update/seconds | not reached (censored) | 281,500 / 16661.497s | 306,500 / 6890.168s |
| P3 max-anywhere | 3/512 | 512/512 | 512/512 |
| P3 fixed-final | 0/512 | 512/512 | 512/512 |
| P4 first>=50 update/seconds | not reached (censored) | 348,500 / 20410.467s | 309,000 / 6933.623s |
| P4 first>=512 update/seconds | not reached (censored) | not reached (censored) | 391,500 / 8605.185s |
| P4 max-anywhere | 2/512 | 450/512 | 512/512 |
| P4 fixed-final | 0/512 | 448/512 | 512/512 |
| P5 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P5 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P5 max-anywhere | 2/512 | 2/512 | 15/512 |
| P5 fixed-final | 0/512 | 0/512 | 7/512 |
| P6 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P6 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P6 max-anywhere | 2/512 | 2/512 | 2/512 |
| P6 fixed-final | 0/512 | 0/512 | 0/512 |
| P7 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P7 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P7 max-anywhere | 4/512 | 4/512 | 2/512 |
| P7 fixed-final | 1/512 | 0/512 | 1/512 |
| P8 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P8 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P8 max-anywhere | 3/512 | 2/512 | 3/512 |
| P8 fixed-final | 0/512 | 0/512 | 0/512 |
| P9 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P9 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P9 max-anywhere | 3/512 | 2/512 | 2/512 |
| P9 fixed-final | 0/512 | 0/512 | 0/512 |
| P10 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P10 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P10 max-anywhere | 2/512 | 3/512 | 2/512 |
| P10 fixed-final | 0/512 | 0/512 | 0/512 |
| P11 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P11 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P11 max-anywhere | 2/512 | 2/512 | 2/512 |
| P11 fixed-final | 0/512 | 0/512 | 0/512 |
| P12 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P12 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P12 max-anywhere | 2/512 | 2/512 | 3/512 |
| P12 fixed-final | 1/512 | 0/512 | 0/512 |
| P13 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P13 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P13 max-anywhere | 2/512 | 3/512 | 2/512 |
| P13 fixed-final | 0/512 | 0/512 | 0/512 |
| P14 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P14 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P14 max-anywhere | 3/512 | 2/512 | 3/512 |
| P14 fixed-final | 0/512 | 0/512 | 0/512 |
| P15 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P15 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P15 max-anywhere | 2/512 | 3/512 | 3/512 |
| P15 fixed-final | 0/512 | 0/512 | 0/512 |
| P16 first>=50 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P16 first>=512 update/seconds | not reached (censored) | not reached (censored) | not reached (censored) |
| P16 max-anywhere | 2/512 | 2/512 | 2/512 |
| P16 fixed-final | 0/512 | 0/512 | 0/512 |

Fixed-final contiguous perfect prefix=1; terminal P16=0/512; suffix7-restricted=1/4096.


## 4. Validation, identities and reproduction

CPU-only read-only audit on **lambda01**, using project `.venv` Python3.12.3 and `CUDA_VISIBLE_DEVICES=''`:

- The registered analyzer passed verification of **184 source/artifact bindings and389 dependency files**, including current bytes and committed relative-path blobs, then produced the exact both-baseline tables above. Registration SHA256: `57f9a7c8c76d9cf975803ae4b081e1793dfc0b304c2938a810927d5582ab0b1f`. Both run configs bind the prelaunch commit502329ab and identical registration/pool identities.
- Watcher `--once` returned **0: DONE X10b higher-LR Muon pair run**. Independent audit checked terminal==DONE, exit0/no error, result hashes, safety-ceiling final==last validation, all1,001 validation steps and denominators/position totals/prefixes/suffix totals, **all1,000,000 consecutive finite training rows**, exact multiplier schedules at every update, elapsed/timing sums and retained entry/best/final checkpoint byte hashes. Both observed peaks are **298,329,600 allocated bytes**. This is receipt/byte verification, not checkpoint deserialization or prediction re-evaluation.
- Existing source-named CPU regressions: **26 passed,6 GPU cases deselected** in32.66s. No tests or implementation code added/changed; no GPU test was rerun. The source-bound prelaunch numerical tests remain separately identified in the launch report.
- Documentation-only change: this report plus a marked one-paragraph addendum in [the optimizer decision report](docs/OPTIMIZERS-2026-report.md#addendum--completed-stage-1-lr-scale-correction-x10b-2026-10-21); the reviewed body/preregistration is preserved. No production/W3/W4 or distributed/resilient behavior is changed or tested; requirement-matrix and scheduler evidence are not applicable to this standalone local readout. Independent acceptance review remains required.

Hashes identify local raw receipts/checkpoints, **not a claim that mutable streams or binary checkpoints were committed/pushed**. Historical baseline result/validation hashes are frozen in `C/baselines/` and tabulated in X10. New-arm identities:

| Arm | Artifact | SHA256 |
|---|---|---|
| 3x | result.json | `2bd5e1705c107e3e079393bd98b4deb886f575a1a7a9bc591ceb6c1b9e272ba2` |
| 3x | validation.json | `68e0334d04cb15461878c3f6cb25c2daea3d82f81c26ee93554b50cf28371e4d` |
| 3x | training.jsonl | `99f3008908a49ea3b7452f1c7ab966636c2fd3c2b9ef5b83de521c93065da529` |
| 3x | step_500000.pt | `119b2315c302225f714b9a6b80c500b647681dd7c431d76482b0042b70b6ddfb` |
| 10x | result.json | `407f80c6f0b8afef90f361e8471cd8a5e66339964060fdcd826e8c58f20c72f4` |
| 10x | validation.json | `cec2a8021bcb8fd62c199b11bf9440077dd08b4a23a693ba4004e5a95791be79` |
| 10x | training.jsonl | `7d8edbe8c7646e1533be1a939301bbafa2e737e85ad93afa977aa9b5dfc0a010` |
| 10x | step_500000.pt | `f7eebcd09ef6b3bda2f72f5f065698e73ad1e4fe683b7aede21b92a3d4226df7` |

Commands run from repository root (temporary readout/audit outputs are not deliverables):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.invertible_stage10b_control analyze --output /tmp/X10b-registered-readout.md
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/watch_invertible_stage10b_muonlr.py --once
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/audit_x10b_growth.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_invertible_stage10b_muonlr_8m.py tests/test_invertible_stage10_optimizers_8m.py -k 'not gpu'
git diff --check
```

### Receipt audit source

For independent reproduction, save this as `/tmp/audit_x10b_growth.py` and run the command above. It requires the local raw artifacts and original dependency environment; no model is loaded or evaluated.

```python
import hashlib, json, math
from pathlib import Path
from experiments.expressivity_tasks import run_invertible_stage10b_muonlr_8m as run

root = Path('experiments/expressivity_tasks/results_invertible_stage10b_muonlr_8m')
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
reg = run.verify_registration(root)  # Current/committed source and dependency bytes.
summary = {}
expected = {
    'muonlr3x': [512]*4 + [12,0,0,0,0,1,1,0,0,0,0,0],
    'muonlr10x': [512,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0],
}
for arm, counts in expected.items():
    name = f'{arm}_seed0'
    p = root / name
    rows, res, cfg = (read(p/f) for f in ('validation.json','result.json','config.json'))
    term = read(root/f'receipts/{name}/terminal.json')
    assert term == read(root/f'markers/{name}.DONE')
    assert term['exit_code'] == 0 and term['error'] is None
    assert term['result_sha256'] == sha(p/'result.json')
    assert res['status'] == 'complete' and res['total_steps'] == 500000
    assert res['stop_reason'] == 'safety_ceiling' and res['final'] == rows[-1]
    assert cfg['registration_sha256'] == sha(root/'registration.json')
    assert cfg['registration'] == reg and cfg['seed'] == 0 and cfg['arm'] == arm
    assert read(p/'data_receipt.json')['array_sha256'] == reg['pool_array_sha256']['0']
    assert read(p/'data_receipt.json')['archive_sha256'] == sha(p/'frozen_validation.npz')
    assert [r['step'] for r in rows] == list(range(0,500001,500))
    for r in rows:
        pos, agg = r['positions'], r['aggregate']
        assert len(pos) == 16 and all(v['count'] == 512 and 0 <= v['correct'] <= 512 for v in pos)
        assert agg['count'] == 8192 and sum(v['correct'] for v in pos) == agg['correct']
        assert agg['terminal_correct'] == pos[-1]['correct'] and agg['terminal_count'] == 512
        assert agg['post_suffix7_correct'] == sum(v['correct'] for v in pos[8:])
        assert agg['post_suffix7_count'] == 4096
        assert all(math.isfinite(v['loss']) for v in pos) and math.isfinite(agg['loss'])
        assert r['perfect_prefix_depth'] == next((i for i,v in enumerate(pos) if v['correct'] != 512),16)
    assert [v['correct'] for v in rows[-1]['positions']] == counts
    first = lambda i,t: next((r['step'] for r in rows if r['positions'][i]['correct'] >= t),None)
    assert all(first(i,50) is None for i in range(4 if arm == 'muonlr3x' else 1,16))
    opt_total, update_total, elapsed = 0., 0., 0.
    with (p/'training.jsonl').open() as stream:
        for n,line in enumerate(stream,1):
            a = json.loads(line)
            assert a['step'] == n and math.isclose(a['lr'],run.learning_rate(arm,n),rel_tol=1e-14)
            assert all(math.isfinite(v) for k,v in a.items() if v is not None)
            assert a['elapsed_s'] > elapsed and a['optimizer_seconds'] >= 0 and a['update_seconds'] >= 0
            assert (a['full_update_rms'] is not None) == (n % 500 == 0)
            elapsed = a['elapsed_s']
            opt_total += a['optimizer_seconds']; update_total += a['update_seconds']
    assert n == 500000 and math.isclose(opt_total,rows[-1]['optimizer_seconds_total'],abs_tol=1e-6)
    assert res['wall_seconds'] >= elapsed
    cps = read(p/'checkpoints.json')
    for cp in (rows[0]['checkpoint'],res['best']['checkpoint'],res['final']['checkpoint']):
        assert cps[cp['path']] == cp and sha(Path(cp['path'])) == cp['sha256']
    best = next(r for r in rows if r['step'] == res['best']['step'])
    assert best['aggregate']['correct'] == max(r['aggregate']['correct'] for r in rows)
    summary[arm] = dict(final=counts, maxima=[max(r['positions'][i]['correct'] for r in rows) for i in range(16)],
        first50=[first(i,50) for i in range(16)],first512=[first(i,512) for i in range(16)],
        best_step=best['step'],best_counts=[v['correct'] for v in best['positions']],
        wall_seconds=res['wall_seconds'],optimizer_seconds=opt_total,update_seconds=update_total,
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
        hashes={f:sha(p/f) for f in ('result.json','validation.json','training.jsonl')},
        final_checkpoint_sha256=res['final']['checkpoint']['sha256'])
print(json.dumps(summary,indent=2))
print('PASS: X10b sources/dependencies, terminal bindings, 1,000,000 finite consecutive LR rows, validation grids, retained checkpoint hashes')

```
