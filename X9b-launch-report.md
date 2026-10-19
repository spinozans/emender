# X9b — schedule-under-ScheduleFree decomposition cell

**LIVE on card2, lambda01; launch2026-10-19T21:58:33Z.** Registration, operational sources and passing preflight receipts were committed and pushed **before launch** as `39dd6748713361355a615b84f5f9c6189e8b08da`, together with the [completed X9 growth report](X9-growth-report.md). Evidence root (**C/**): `experiments/expressivity_tasks/results_invertible_stage9b_sf_anneal_8m/`. This is a launch report, not a learning verdict. No X6/X7 process, source, lease or checkpoint was changed; all **five previously live trainer identities** remain alive with the same PID/starttime. Historical completed trainer identities were also unchanged.

## Two predictions registered before launch

1. **(i)** If X9b composes (**any P2+ reaches >=50/512**), the **SCHEDULE carries the unlock and SF was masking it**: external scheduling suffices even while SF averaging is retained.
2. **(ii)** If X9b stays flat like X7 (no P2+ reaches50), the **SCHEDULE-FREE CLASS is the binding-structure hypothesis**: its averaging/anchor dynamics gate basin entry; the remaining **AdamW+constant** cell, queued for the next free card, is the confirming follow-up. **That cell was not launched here.**

These are registered hypotheses, not established mechanisms. **One seed, one depth, one regime.** X9b retains X7 betas(.9,.95), so X9b versus X7 isolates the external schedule within SF. X9 versus X9b still changes **beta2(.999 versus.95)**, optimizer class and decay semantics: a negative X9b alone does not prove averaging specifically caused failure or rule out beta effects. The remaining cell and appropriately matched controls are needed before unique attribution. Partial acquisition is reported literally, not relabeled a full T16 solution.

## Recipe and actual LR

Scratch **E97 bf16 T16**, seed0,9,900,800 parameters; exact X7 initial model and frozen512-example validation pool verified. Uniform mean target CE, batch32, clip1, fresh procedural training; no ladder, auxiliary carry supervision, restart, stagnation rule or threshold stop. Only the **500,000-update safety ceiling** stops the run. Every500 updates, including baseline/exit, report unweighted P1–P16 joint/x/y counts and CE, aggregate/suffix/terminal metrics and perfect-prefix depth. Parameters are fp32; bf16 autocast and legacy recurrent policy/sequential Triton evaluation match X7.

**AdamWScheduleFree**, betas(.9,.95), eps1e-8, decay.01, **internal warmup_steps=0**. Explicit per-update external LR: `1e-3*step/5000` at updates1..5000 (first2e-7, peak1e-3), then `1e-5 + .5*(1e-3-1e-5)*(1+cos(pi*(step-5000)/495000))`, ending **1e-5 at500K**. Each update logs both requested LR and the optimizer's **actual `scheduled_lr`**, preventing a silent double warmup. SF averaging is **retained**, not emulated or removed: `.train()` presents y, `.eval()` presents averaged x; unchanged r=0, weight_lr_power=2 and lr_max weighting (the averaging weight follows the retained maximum LR, not declining instantaneous cosine LR). Checkpoints explicitly store **ScheduleFree eval x**, optimizer y/z-related state and RNGs. Decay-at-y remains the X7 semantics; integrated decay changes with the schedule.

## Verified execution and bounded evidence

Detached worker **2138210** (starttime251310149), **PPID1/SID2138210**; trainer **2138313** (starttime251310203) belongs to that worker/session. Standalone exclusive card2 lease matches worker identity. No WorksGood service/dispatcher/completion machinery was used. The launcher fail-closes unless registration, operational source bytes, installed SF source and passing preflight receipts match preregistered hashes and committed HEAD.

`C/receipts/verify_launch.py` passed at **21:59:18Z**: source/dependency identities, exact prelaunch commit, frozen pool/archive, entry hash and exact X7 scratch weights, empty entry optimizer state, averaged-checkpoint semantics, trained SF z state, detached process/lease identity and the prior live trainers. Snapshot: **1,331 finite consecutive updates**, last loss24.868038177, requested/actual LR**2.662e-4**, elapsed34.012s; latest validation1000. Baseline joint P1–P16 is`1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0`, aggregate2/8192, CE250.976102829, exactly X7's baseline. These are startup observations, not a growth result or runtime guarantee.

`launch_snapshot/` freezes finite training prefix, config/data, validation, checkpoint identities and the frozen pool. `full_launch/` freezes operational code, installed SF implementation with its license, package list, launch commit/diff and GPU/disk evidence; `full_launch_manifest.json` hashes it. Live streams/checkpoints are not committed mid-write. Immutable entry plus aggregate-best and latest/exit follow X6D's existing hash-manifest pruning below400GiB free (200,760,844,288 bytes available before launch). No historical pruning or restart was performed. The verifier only hashes/loads this run's retained entry and trained checkpoint plus the immutable X7 entry; it does not re-evaluate predictions.

## Supervisor watcher: arming required

**Arming is supervisor-owned, not performed by the worker.** Supervisor was notified after verified launch. Native watched command:

```bash
/home/erikg/emender/.venv/bin/python /home/erikg/emender/scripts/watch_invertible_stage9b_sf_anneal.py --interval 300
```

Independent `--once` at21:59:34Z returned **2 (healthy/pending)**; receipts retain the log and exit code. While pending it checks worker/trainer PID+start identities; on terminal completion it requires matching atomic DONE/terminal records, successful exit, exact result SHA256 and complete500K/safety-ceiling result. FAIL/dead/reused identities return nonzero. This is not host-reboot recovery. Supervisor should arrange native wake on exit, then audit completion and write the X9b growth report against predictions(i)/(ii); do not infer a verdict from baseline.

## Validation and scope

**Six X9b tests passed under an exclusive card2 lease** (19.99s), including real bf16 finite backward, actual SF update/train/eval and per-position validation. CPU regression: **10 passed,2 GPU skips** across X9/X9b (21.77s). Tests cover exact schedule endpoints/full monotonicity, actual scheduled LR with internal warmup disabled, retained averaging and peak-LR weighting, exact X7 scratch model, safety-only continuation despite perfect mocked validation, immutable-entry retention, SF eval-x checkpoint state, and watcher pending/DONE/FAIL/dead-identity/mismatched-terminal behavior. Initial CPU attempt exposed empty SF state dictionaries for parameters unused by the synthetic forward; its assertion was corrected to check trained states, and all final tests passed (initial log retained). Python compilation, shell syntax and operational-source diff checks pass. Verbatim archived upstream SF code retains original whitespace.

Local single-card expressivity only; `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` and gap matrix **R16 evidence discipline by analogy only**, not production/resilient DiLoCo or Slurm Partition/QOS/systems-scale acceptance. Independent review remains supervisor-owned. **Next step: arm the native watcher; no other cell or scale launch.**
