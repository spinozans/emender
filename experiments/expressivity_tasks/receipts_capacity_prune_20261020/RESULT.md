# Emergency capacity prune result

Implemented the task with the explicit stage2b emergency override documented
in AUTHORIZATION.md. No training code or pre-existing evidence files changed.

| Target | Files removed | Bytes freed | GiB freed |
|---|---:|---:|---:|
| stage2b | 321 | 826057416819 | 769.325920 |
| stage5c_growth_8m | 1904 | 226144856978 | 210.613810 |
| stage3_8m | 158 | 6074627290 | 5.657438 |
| stage2, stage2c | 0 | 0 | 0 |
| stage5b_growth_8m, stage5_instrument | 0 | 0 | 0 |
| stage4_13b, stage6_truecur_8m | 0 | 0 | 0 |
| 11 inactive Emender pi-bash logs (gzip) | 11 originals replaced | 1168629 net | 0.001088 |
| **Total** | **2383 checkpoints + 11 logs** | **1058278069716** | **985.598256** |

Validation: six safety unit tests, independent manifest-entry SHA256 and
original-ledger reconciliation, per-file execution reconciliation, byte-identical
non-checkpoint evidence in every inactive root, preserved stage2b
best/final/matched/stops, preserved X3 best/final and full one-step diagnostics,
and fresh retained best/stop/X5c payload hashes. All compressed logs passed
round-trip content SHA256 checks; the generated gzip files are mode 0600.
The large bioinformatics log named as an example in the task was left untouched.

The authoritative final free-space and df evidence is in validation.json and
validation_output.txt. More than 500 GiB remained free after validation.

Residual limitations: removed payload SHA256s use the original checkpoint
ledgers, not a new full rehash before deletion. X5b/instrument schema ambiguity
caused full retention; the live X6 root was retained without external pruning,
and its training logs continue to advance. Archived X2 alignment weights were
also left untouched. No nvme1/external X4 weights were traversed or modified.

Commands (on lambda01, not Frontier):

```
.venv/bin/python experiments/expressivity_tasks/receipts_capacity_prune_20261020/prune.py
.venv/bin/python -m unittest discover -s experiments/expressivity_tasks/receipts_capacity_prune_20261020 -p test_prune.py -v
.venv/bin/python experiments/expressivity_tasks/receipts_capacity_prune_20261020/prune.py --apply
.venv/bin/python experiments/expressivity_tasks/receipts_capacity_prune_20261020/compress_project_logs.py
.venv/bin/python experiments/expressivity_tasks/receipts_capacity_prune_20261020/validate.py
```

Do not rerun the apply command against these receipts. The durable manifests
and their execution logs are the record of this one attended prune, not a
standing authorization to remove future checkpoints. Next step: independent
review of the committed receipts, and resume capacity-blocked work under the
operator's control.
