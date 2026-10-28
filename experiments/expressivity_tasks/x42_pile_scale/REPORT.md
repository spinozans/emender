# X42 part 2 — corpus construction in progress

The 24 GiB document-ordered corpus build is **running, not complete**. The
detached builder is PID 793830, reparented to PID 1, using only the required
`/home/erikg/emender/.venv/bin/python`. See
`receipts/part2_corpus_launch.json` for the timestamped process and progress
snapshot. It streams the pinned uncopyrighted Pile substitute; it is not the
complete original Pile. No pilot or search was launched.

The source starts at document index zero and advances without shuffling. The
stored train and validation streams preserve that order after exact deduplication
and the registered document-hash split. Thus the large source interval is
contiguous, while each individual split excludes documents assigned to the other
split and duplicate documents. The registered 1M-character cap and final-token
budget truncation remain explicit. The document index records source positions,
content hashes, split, token offsets, lengths and final truncation. The completed
manifest will contain source-interval and per-split boundaries, per-split document
and token counts, split hashes, document-index hash and tokenizer identity/hash.
No random fragment collection is substituted for a coherent source stretch.

Construction now emits atomic `progress.json` snapshots every 1000 retained
documents, marks iterator/tokenizer failures, checks the dedup database size and
free disk space, and enforces a 90 GiB build-directory ceiling. Tokens are capped
at 24 GiB. No model files are emitted. The access sample remains separate.

Validation: `receipts/part2_tests.txt` records **12 passed** (the original 11
checks plus a corpus failure check). The corpus test now verifies document
ordering, source boundaries, split document counts, contiguous token offsets,
and completion status. These are fixture checks, not full-corpus validation.

The initial shell-background launch exited before startup; it was replaced by
an explicit detached process, and its PID 1 parent and advancing progress were
verified. To monitor the live build from the repository root:

```bash
ps -p 793830 -o pid,ppid,etime,pcpu,rss,args
cat experiments/expressivity_tasks/x42_pile_scale/data/pile24/progress.json
tail experiments/expressivity_tasks/x42_pile_scale/receipts/corpus_build_console.txt
du -sh experiments/expressivity_tasks/x42_pile_scale/data/pile24
```

Completion requires `data/pile24/manifest.json` with `complete: true` and a
subsequent verification of its hashes/counts. A live PID or an advancing progress
file alone does not establish completion. Do not restart into this existing
directory: corpus construction has no resume support yet.

Residual work: full build completion and attestation; leased GPU numerical,
gradient, replay, throughput and measured-memory validation at both scales;
training checkpoint/resume and pilot-shape kill/resume evidence; heads-spectrum
protocol amendment before search. The old E97 methodology receipts were read:
their ranking used mean training trajectory loss over 15-minute candidates,
p50k_base and population 8. X42's existing fixed-exposure held-out objective,
GPT-2 tokenizer and population 12 are explicit protocol differences, not an
exact reproduction. No search protocol was changed in this part. X40 gates
pilot/search runs and does not block these outstanding engineering checks.

`REPORT.md` did not exist on entry; `RECEIPTS.md`, the 11-test infrastructure
suite and the 256-document sample receipts were read instead. No GPU was
acquired and no resilient DiLoCo behavior was changed. The corpus build is the
long pole; remaining adapter/resume work is deferred to part 3 without claiming
those outcomes.
