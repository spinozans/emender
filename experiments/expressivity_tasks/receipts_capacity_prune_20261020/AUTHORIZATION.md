# Emergency capacity prune — 2026-10-20

Authority: `/home/erikg/emender-scratch/fix-fanout/taskCAPACITY-prune.md`,
plus two live supervisor rulings received during this attended execution on
2026-10-20 (UTC; execution manifests supply registration times).

## Supervisor override, first ruling

The stage2b 988G root was already pruned to the prior required set. The worker
paused rather than delete required weights. The supervisor authorized emergency
narrowing because the root volume was 100%, a live run had already failed with
`Errno 28`, and W3 production arms were blocked. All non-checkpoint evidence
must survive. No nvme1/X4/1.3B weights or other volumes may be touched.

## Supervisor clarification, controlling ruling

The worker pointed out that the initial override's reference-to-any-manifest
clause would retain every checkpoint. The supervisor explicitly approved:

> Routine inventory ledgers (checkpoints.json, per-validation checkpoint
> records, historical pruning manifests) are RECORDS of existence, not
> independent pin evidence.

Retain (a) every run's final and terminal-best checkpoint, (b) legacy/matched
points and broadcast/own stops from the prior manifest's named classes, and
(c) checkpoints explicitly hash-cited by downstream reports, reviews, or
evaluation receipts. Delete historical best-chain members referenced only by
routine inventories. Preserve every deleted filename, size, and SHA256 in the
new manifest. Ambiguity always means retention.

The overridden required-set document is:
`experiments/expressivity_tasks/results_invertible_stage2b/receipts/postcompletion_capacity_prune.json`.
Its bytes are preserved and checked by the new stage2b evidence snapshot.
The supervisor reduced the practical capacity gate to >=200G with margin
(>=300G preferred), without authorizing any broader deletion. Execution
nevertheless exceeded the original >=500G free-space target.

## Other roots and temporary logs

Other roots follow the original task, not the stage2b exception. Historical
best chains remain protected there. All one-step X3 diagnostics were retained
in full, along with every X3 run's best/final checkpoints. X5c independent
receipt references were retained (plus latest checkpoints conservatively).
X5b and stage5_instrument validation schemas did not match the simple chain
selector: their entire checkpoint inventories were retained as ambiguous.
Stage6 is active and was retained in full. Stage4 has no local weights; external
checkpoint storage was neither traversed nor modified.

The task's suggested `/tmp/pi-bash-cab7545f335f075e.log` is actually 18.9GB of
bioinformatics output (`1gbwt`, `1khash`, `source_gap_bp`,
`oriented_continuation`), so it was **not compressed or modified**. Only 11
positively identified inactive Emender pi-bash logs were gzip-compressed.
Unknown, mixed/bioinformatics, open, large/unverified, or uncertain logs were
retained. Each compressed log was round-trip SHA256 checked before unlinking
its original. No receipts were compressed.

## Execution and validation

- `*.json`: per-target plans durably written before the first unlink, containing
  every removed file, original checkpoint-ledger SHA256 and size, plus SHA256
  of its canonical manifest entry (sorted compact JSON without the hash field).
- `*.execution.jsonl`: per-file deletion/compression receipts, flushed/fsynced.
- `reference_scan.json`: hashed source inventory of repository and fix-fanout
  report/review/manifest pin scans and explicit routine-record exclusions.
- `dry_run/`: pre-execution inspection only; no deletions were made by that run.
- `prune_at_execution.py`: the execution source. `prune.py` additionally refuses
  all existing manifests before any writes and refreshes active-process checks
  per root; it is not a rerunnable instruction to delete additional weights.
- Removed payload SHA256s are from original checkpoint ledgers, not newly
  rehashed deleted payloads. The ledger evidence survives byte-identically.
- `validate.py` independently checks canonical entry hashes, original ledgers,
  execution logs, retained required classes, unchanged non-checkpoint evidence,
  gzip round trips, and the free-space target. Retained terminal-best/stop
  weights in stage2b/X3 and all retained X5c weights are freshly rehashed.
- Stage6 training evidence advances naturally; its unchanged-evidence check is
  explicitly inapplicable because this task deletes nothing in that live root.
- Execution occurred on `lambda01`, using the repository `.venv/bin/python`,
  not Frontier. No resilient DiLoCo behavior, scheduler jobs, training code,
  AGENTS.md, or CLAUDE.md was changed.
