# X9 — AdamW warmup/cosine regime swap: launch report

**LIVE: one E97 bf16 run on card 2, lambda01.** Registration and operational sources were committed **before launch** as **`8e6d1393`**; launch **2026-10-19T18:23:54Z**. Evidence root (**C/**): `experiments/expressivity_tasks/results_invertible_stage9_anneal_8m/`. This is launch evidence, not a growth verdict. All six previously live X6/X7/X8 trainer PID/start identities were verified unchanged; no historical run, source, lease or checkpoint was modified.

## Approved scope correction, registered before launch

Code inspection found X7 uses **AdamWScheduleFree**, not ordinary AdamW. The supervisor explicitly amended X9 from a schedule-only probe to a **REGIME SWAP**: ordinary **AdamW**, standard betas **(0.9, 0.999)**, explicit warmup/cosine, versus X7's **constant-lr AdamWScheduleFree, betas (0.9, 0.95)**. X6 best-vs-final degradations already occurred **under ScheduleFree**, weakening the original plain-oscillation explanation. It would be misleading to label this run a clean annealing-only comparison.

Weight decay retains the closest practical **nominal 0.01**, eps1e-8. This is not exact effective equivalence: ScheduleFree adds decay at y after variance normalization, before its averaged y/z updates; ordinary AdamW decouples parameter decay. Integrated shrink also changes with LR. Optimizer class, standard-beta change and schedule are a single registered intervention bundle; decomposition is an explicitly unresolved follow-up, not a result of this run.

## Registered predictions and honest asymmetry

1. **(a)** If the optimizer regime was structural, ordinary AdamW plus anneal can consolidate a partial-carry circuit once found. Any driver position **P2 or deeper reaching >=50/512** counts as material against X7's flat `512,2,0,...`; report exact P1–P16 columns.
2. **(b)** If still flat, this tested optimizer-regime bundle fails to unlock basin entry. Fixed-lr/ScheduleFree is weakened as the explanation, completing the planned 8M probes in this series; remaining interrogations are **scale** (1.3B deep-first and ladder-then-deep) and **supervision-schedule redesign** (the proven unlock, whose current wean fails at depth4+). A single seed cannot establish that all optimizers fail.
3. **Consolidation versus entry:** anneal plausibly helps **CONSOLIDATION**, not **basin ENTRY**. A small-P2-only or partial-P2 outcome is **consolidation-only evidence**, not proof of a deep entry unlock.
4. Anneal versus ScheduleFree removal versus beta change remains unresolved; no schedule-only causal claim is authorized.

These predictions, amendment, source hashes and preflight hashes are in `C/registration.json`. Launch rejects source, registration or preflight bytes differing from committed HEAD. No other arm was launched.

## Recipe

E97 **9,900,800 parameters**, dim256/four layers/MLP ratio2, p61/vocabulary3721, **T16 from scratch**, seed0, empty optimizer. **Uniform mean cross entropy** over driver targets (X8 weighting is not used). Batch32, gradient clip1, fresh procedural training batches; **no ladder, carry supervision, stagnation rule or threshold stop**. Only the **500,000-update safety ceiling** stops training.

Actual per-update LR is set before `opt.step()` and logged: update1 **2e-7**, linear `1e-3 * step/5000` through update5000 **1e-3**, then `1e-5 + 0.5*(1e-3-1e-5)*(1+cos(pi*(step-5000)/495000))` through update500000 **1e-5**. No optimizer-internal warmup and no ScheduleFree train/eval averaging remain. Checkpoints correctly label **ordinary AdamW parameters**, with optimizer and RNG state.

Everything else retains X7 bf16: fp32 parameters, bf16 autocast, legacy recurrent state policy, sequential Triton E97 with `fused_inference=True` at evaluation; live accumulator already fp32. Input18 causally right-pads to32 using the unchanged X3 helper. Frozen seed0 T16 pool equals X7 byte-for-byte by array hashes; archive hash also verified. Every500 updates, including step0/final500K, a complete512-example validation reports P1–P16 joint/x/y counts and CE, aggregate/suffix/terminal metrics and consecutive perfect-prefix depth. Training never samples validation.

## Verified launch and retention

`C/receipts/verify_launch.py` / `launch_verification.json` check registered source/dependency hashes, prelaunch commit, frozen arrays/archive, entry checkpoint hash, identical X7 initial model weights, empty AdamW state, finite consecutive T16 updates and exact actual-LR schedule. Detached worker **2311198**, **PPID1**, **SID2311198**; trainer **2312873** has that worker as parent/session, and exclusive card2 lease matches PID/starttime. No WorksGood service, dispatcher or worker machinery was started; only the existing standalone GPU lease script was reused.

Snapshot **2026-10-19T18:24:08Z**: update181, LR **3.62e-5**, loss185.4956; **53.26 updates/s** measured from update100, roughly **2.61h remaining** at that early rate. This is not a completion guarantee or learning finding. Step0 joint-correct P1–P16: **`1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0`**, aggregate2/8192, unweighted CE250.976103, perfect-prefix0, matching X7 bf16 initialization.

`launch_snapshot/` freezes config/data/validation/checkpoint indices, frozen pool and finite training prefix. `full_launch/` archives operational sources, installed FLA dependency bytes **with upstream MIT license**, ordinary AdamW implementation, ScheduleFree control implementation, package list, base commit/diff and GPU/disk snapshots; `full_launch_manifest.json` hashes them. Live training files/checkpoints are not committed mid-write.

Reuse X6D retention: **immutable entry**, aggregate-best and latest/exit. Below400GiB free (about262GiB at launch), prune all other checkpoints; manifests retain removed paths and hashes. Final500K is the exit/latest. Synthetic tests exercise the actual pruner with changing bests; `retention_audit.json` and `retention_snapshot/` capture real retained entry/hash evidence. No historical deletion.

## Watcher and next deliverable

**Native supervisor watcher armed (PID2653261, started2026-10-19T18:25:24Z, first healthy poll18:25:25Z)**, supervised process **`proc_428f`**. `C/receipts/watcher_armed.json` records the verified identity and parent-native wake confirmation. Command: `scripts/watch_invertible_stage9_anneal.py --interval 300`. The parent confirms exit (DONE or failure) wakes its session natively. Independent `--once` returned **2 (healthy/pending)**. It validates worker/trainer PID/start identity while pending, detects failure markers/dead or reused processes, and returns zero only after the successful DONE receipt and result exist. Workers atomically publish terminal receipts and DONE/FAIL markers. It is not host-reboot recovery; re-arm after reboot.

**On native watcher wake:** verify terminal/source/checkpoint identities and write the growth report against (a)/(b), with exact P1–P16 trajectories and positions reaching50/512. Preserve the consolidation-versus-entry asymmetry and regime-bundle attribution limitation. Do not infer outcomes from launch baseline.

## Validation / scope

**203 CPU tests passed, five GPU skips**, plus **all six X9 tests passed under the exclusive card2 lease**, including full-model bf16 finite uniform-loss backward and complete per-position evaluation. New tests cover schedule endpoints and all-update monotonicity, actual AdamW class/betas/decay/update, exact scratch X7 model identity, safety-only continuation despite perfect mocked validation, optimizer checkpoint semantics, immutable-entry retention with real prune manifests, and watcher pending/DONE/FAIL/dead-identity paths. Exact commands/output are retained in `C/receipts/tests_{cpu,prelaunch}.txt`, exit0 in `tests_exit_code.txt`. Initial broad CPU attempt exceeded its120-second command budget; the360-second rerun passed in121.31 seconds. Python compilation, `bash -n`, operational-source whitespace and launch verification pass. Verbatim upstream Torch license and ScheduleFree implementation archives retain their original trailing whitespace; the archive-excluded diff check passes.

Architecture authority reviewed: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` and `docs/RESILIENT_DILOCO_GAP_MATRIX.md`; **R16 evidence discipline only**, no resilient DiLoCo, Frontier Slurm Partition/QOS or scale acceptance claimed. Independent reviewer gate remains supervisor-owned. Next scientific deliverable: growth report on watcher wake.
