# X17: carry-supervised T64, one registered slow wean (pre-launch)

Standalone E97 from scratch, seed0, deep-first T64, p61, batch32. Four dim256
layers; exact X10 legacy-bf16 training recurrence at train AND evaluation,
fp32 parameters/bf16 autocast. Stage10-pattern code paths only: no curriculum
runner, optimizer adapter, production or distributed/resilient training changes.
No Frontier submission; resilient DiLoCo requirement IDs are not applicable.

## X6 Arm B mechanism and explicit supervisor format ruling

Preserve EXACT X6 Arm B: fixed carry slots before **P2..P64 only**, no extra
slot before P1. Supervisor approved this explicitly before preregistration:
changing the X6 format would contaminate the mechanism-transfer claim.
129 input tokens right-padded to144 by X3 padded_forward; gather logits back to
original66 positions. Coordinate headers and driver positions remain unchanged.
Input-only neutral3721 has its own learned256-vector, initialized exactly as X6;
input vocab3722, output vocab3721, total9,901,056 parameters. Before each P2+
driver the slot independently contains the TRUE **previous** joint-state token
with probability p, otherwise neutral; never current/future targets. At eval,
**ALL slots ALWAYS neutral**, same input format, target-independent, no feedback.
P1 has no supervised slot; its learning is the free diagnostic, not composition.

## The ONE registered wean

p(update) = 0.5 * (1 - update / 1,000,000), globally across the full1M ceiling.
No per-stage re-crutch/re-wean cycling, no resets, no adaptive schedule. p starts
at0.5 (update0), is0.4999995 at update1 and exactly0 at update1M. Actual p logged
EVERY update, and attached to every validation row (eval p separately fixed0).
Fresh procedural numpy.default_rng(0) batches; carry Bernoulli draws share that
RNG, exactly as X6, so subsequent batches need not equal cold-run batches.

## Exact X12 AdamW + cosine

Ordinary live parameters; AdamW betas(.9,.999), eps1e-8; ALL parameters wd.01,
including neutral embedding; clip norm1. Uniform masked-target CE across P1..P64.
Updates1..5000 linear LR warmup to1e-3; cosine to1e-5 at1M (1M denominator, not
X10's500K). Actual LR logged every update. Fixed1M safety ceiling ONLY: no
stagnation, threshold, all-perfect, curriculum, or wall-time stop. Failures/OOM/
disk/process death are infrastructure censoring, never scientific zeros; no
automatic retries or restart. Checkpoints: entry, aggregate-best, latest/final
(at most3 plus atomic staging); deletion hashes. No best-checkpoint selection
for the primary fixed-final readout.

Frozen exact historical seed0 T64 validation512, hashed pre-launch and verified/
archived at run start, never sampled for training; no test scores. Neutral eval
at update0 and every500 through1M; all64 position counts/512 and CE, first50 and
first512 steps, latest/final perfect prefix and maximum single-validation prefix.
Independent per-position maxima are not a jointly acquired prefix.

## Predictions registered verbatim BEFORE launch

(a) WEAN-TRANSFER: the run acquires carry positions materially (any P2+ >=50/512 during or by end) with the crutch annealed to zero - examples-of-working training transfers to the no-crutch regime at the horizon where cold learning fails; the fix works and the paper gets its training-technology result.

(b) CRUTCH-DEPENDENT-AT-DEPTH: composition appears while p is high but decays as p anneals (X6's cycling signature, flattened) - the carry remains crutch-bound at T64 and the wean problem persists at long horizons.

(c) The P1 diagnostic rides free: does the supervised carry slot also ignite P1-like visible learning earlier than Muon/AdamW cold runs? Logged either way.

Interpretation discipline: neutral evaluations while training p is high are not
true-feedback evaluations. Report transient P2+50 separately from survival at
p=0/final1M; an early crossing alone cannot establish zero-crutch transfer.
If composition never appears, do not force either(a) or(b). Compare P1 timing
with archived cold histories only, naming horizon/LR differences for Muon.

## Fail-closed card-wait launch and native supervision

Poll GPU broker list-free every60s; take FIRST broker-grantable card. No lease
or GPU model transfer/probe while waiting. Broker errors/malformed replies never
admit a card. Worker atomically acquires that card --no-wait, fails closed on a
lost race, and never retries a failed run. Worker heartbeats/releases only its
own lease; host/PID/proc-starttime identities guard supervision. Registration and
all bound source/dependency bytes must match worktree AND committed HEAD before
arming, card admission, worker lease, and trainer start. No WorksGood service.

Native watcher is detached and writes GROWTH_REPORT.md each wake/terminal event:

```
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.watch_invertible_stage17_carrywean --interval 900 --report
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.watch_invertible_stage17_carrywean --once --report
```

Exit0=hash-bound successful1M/p0 terminal; exit1=failed/missing/dead process or
invalid terminal receipt; exit2=pending (--once). Partial histories remain
right-censored; predictions stay unresolved until sufficient terminal evidence.
