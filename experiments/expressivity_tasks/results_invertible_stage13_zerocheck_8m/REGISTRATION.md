# X13 v2: audit FIRST, then exact-X12 E97 T64 seed replication

The operator's v2 task supersedes the earlier zero-check task. X12's receipted
E97 wording is retained, but its interpretation was SUSPENDED pending STEP 0.
NO state/memory/mechanism claims are authorized by this investigation.

## Completed STEP 0 (before registration or replication training)

1. X10 AdamW aggregate-best champion, update468000, SHA256
   `0501a81ff50e34017a973f400eec5975faaf5bd9d84df64d6fa84b51ed6c9fa7`,
   through stage12's E97 model builder and actual loader/padder on the identical
   seed0 frozen T16 validation pool: P1-P4 must retain near512. The registered
   broken example is P1<400/512; any trained position<400 fails closed. The audit
   requires four exactly-perfect positions before allowing replication.
2. CPU T64 builder equivalence, including input tokens, prefix, masks, target
   indices, driver positions and right-padding. Eight frozen examples decoded
   with independent scalar arithmetic. Any divergence is instrument-broken.
3. Only if clean: unchanged champion zero-shot T64 through the trusted stage10
   loader. Report all64 positions, including retention of trained P1-P4 and P5+.

Audit verdict CLEAN: T16 all16 counts match the champion receipt exactly,
P1-P4=512 each; CPU construction agrees; T64 retains P1-P4=512, P5=8.
The historical source hashes match X10/X12 registrations. Receipts are under
`audit/`; the verdict binds the other receipts. No training took place in STEP 0.
This exonerates the audited data path, not every possible implementation detail.

## STEP 1 registered before launch

Only E97 bf16, scratch seeds1 and2, independently `torch.manual_seed(seed)` and
`numpy.default_rng(seed)`. Exactly X12: four layers, dim256, 9,900,800 parameters,
legacy bf16 recurrence, fp32 parameters/bf16 autocast, p61/vocab3721, 64 drivers,
66 input tokens padded to80, batch32. Uniform masked CE; no curriculum, carry
supervision or champion initialization. Ordinary live-parameter AdamW,
betas(.9,.999), eps1e-8, ALL-parameter decay.01, clip1. Updates1..5000 warm to1e-3,
then cosine to1e-5 at1,000,000. The X12 update/evaluator/model/optimizer/schedule
functions are reused unchanged. Actual LR logged every update.

Frozen historical seed-specific T64 val512, hashes checked and arrays archived;
not used as training examples; no held-out/test performance inspected. Validate
P1-P64 counts/CE at0 and every500. Ceiling1M only: no stagnation, metric, all-perfect
or wall-time early stop. Infrastructure failures are censored; no retries/restarts.
Entry + aggregate-best + latest/final retention (at most3 checkpoints plus staging)
with deletion hashes. Best selection is not the primary endpoint.

Original task alternatives: both seeds material anywhere => seed0 lottery;
both zero => replicates. **Explicit supervisor prelaunch ruling** resolves
materiality and the mixed case as follows (before any seed1/2 training):

- **Marginal escape:** ANY P1..P64 >=50/512 at ANY validation. ANY replication
  seed meeting this threshold refutes the seed0 learned-nothing zero as a robust
  outcome: seed0 zero was lottery; re-scope. P1-only escape means learns the
  visible position but not carry at T64; it is NOT evidence of composition.
- **Composition:** separately, ANY P2..P64 >=50/512 at ANY validation.
- If BOTH seeds reach NO position>=50, learned-nothing zero replicates as3/3
  (including historical seed0), for this finite recipe/ceiling only.
- Always report all64 columns and first50/first512 per seed, P1 versus P2+
  separately, fixed-final prefix and independent maxima. Partial zeros cannot
  adjudicate a completed replication. No architecture capacity or mechanism claim.

Registration, audit receipts and all bound operational sources must be committed
before launch; source drift, missing audit, changed hashes or invalid admissions
fail closed before a lease/training. One broker-leased GPU per run; owner PID and
/proc starttime, heartbeats and terminal/DONE/result hash binding. Local standalone
single-card expressivity only, NOT resilient DiLoCo or Frontier; no distributed
requirements are tested or claimed.

## Native supervision

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.watch_invertible_stage13_zerocheck --interval 900 --report
# Nonblocking progress read: exit2=pending, 1=failure, 0=both successful 1M terminals.
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.watch_invertible_stage13_zerocheck --once --report
```

A native supervisor wake must be armed for these workers. On wake, regenerate
`GROWTH_REPORT.md` and update `X13-report.md` with final or explicitly censored
P1-P64 columns and the two distinct threshold verdicts.
