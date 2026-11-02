# Phase-1 receipt layout

`coverage.json`, `max_context.json`, `content_strata.json`, and `samples.json.gz`
retain the conditional interior-token coverage measurement. `*.patterns.gz` and
`*.stdout.gz` retain the exact binary requests and query output, with `*.stderr`
retaining xsa journals. Compression is lossless; filenames in original command
receipts refer to the uncompressed runtime paths. `features.json.gz` has 10,000
sample-aligned rows: 100 empty-context zero rows and two explicitly invalid rows,
not fabricated distributions. `budget.json` separates host process and kernel
costs; `cpu.txt` and `affinity.txt` describe the shared CPU host.

Two UINT64_MAX branch counts caused the initial independent audit to fail.
`initial_validation_failure.txt`, `count_range_failures.json`,
`branch_consistency_failures.json`, and `count_failure_repro.patterns.gz` retain
this failure. `validation.json` passing means the retained measurements are
consistent and the bad rows are rejected; its `query_health` remains failed.
73 located spans/contexts were checked against the read-only Wikipedia bytes.
Raw content overlap is representation dependent and does not establish a source
component fraction; all five centered 256-byte hits were spaces only.

The recorded run used c14_coverage.py before impossible-width rejection was
added. Re-execution with the hardened source now deliberately fails closed at
the branch batch on the currently rebuilt binary. This is not a clean rerun claim.
No sxgc code was changed. After an upstream fix, run c14_coverage.py followed by
c14_supplement.py, independently validate new complete rows, and only then
rebind predictions; do not overwrite these historical failure receipts silently.
The 266 GB materialized rank representation and default Rayon pool are part of
this run's measured machinery, not a demonstrated persistent single-core API.

Audit retained compressed receipts with:

```
/home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/validate_c14.py
```

MANIFEST.json hashes every adopted phase-1 source/report/data receipt except
itself, using repository-relative paths. The program ledger binds the manifest
and principal reports to the exact receipt commit. No .pt artifacts, GPU leases,
large training, held-out inference, or upstream modifications occurred.
