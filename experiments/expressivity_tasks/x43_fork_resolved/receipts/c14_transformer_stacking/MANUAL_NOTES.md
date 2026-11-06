# Manual notes

`contact_supervisor` was searched first by name, then with supervisor/intercom descriptions; it is absent from the active tool catalog. No supervisor reply or approval is implied. Work remains in this attended session, single-threaded, with no WorksGood service or subagents.

Frozen inference used the standalone broker, physical card 2, restricted to eligible cards 2/4. The launcher used Popen(start_new_session=True), DEVNULL stdin and retained stdout/stderr. detachment.json records PPID=1, lease owner/starttime, heartbeat renewal and advancing evaluation targets. This was inference, not training. Card 1, ~/sxgc and /mnt/nvme2n1 were not modified. Model and corpus paths are read-only; no .pt is staged or committed.

The first test invocation failed collection because pytest's default import mode could not resolve package-relative imports. Its full output is retained in tests_initial_collection_failure.txt; the corrected --import-mode=importlib run passed seven numerical and document-boundary tests. No failed result is relabeled green.

Residual risks: one seed; fixed active time with unequal target exposure; no uncertainty estimates; BF16 replay; float32 branch metadata; historical batch-size loss drift; a conditional supported-target oracle is not a total-LM bound. Small post-training fusion residuals do not identify a causal absorption mechanism. Exact foreign-array walks fail and all zero targets remain in totals. No resilient DiLoCo behavior was changed or run, so gap-matrix IDs are inapplicable.

Unrelated pre-existing dirty and untracked files were preserved and excluded from this task's commits.

The intermediate PROGRAM validator failed because the authorized theory revision differed from its old pinned commit. program_validation_before_repin.json retains the failure; final validation requires repinning the new evidence snapshot. Repository-wide diff whitespace checking also found only pre-existing unrelated GROWTH_REPORT.md blank-line changes; task files are checked separately.

Completion: soft transformer P3 passes at 1.2635669715615139 versus 1.2660638292885118 consulted and 1.299507807414495 dense+soft. The independent validator checked all 434,204,084 positions: 224 probability/denominator replays, 40 paired aggregates, maximum finite-sum error 7.450580596923828e-09. Exact-walk failures remain infinite. The figure includes all twelve measured dense/consulted/soft/literal-walk rungs, including signed negative conditional differences from the array, and labels every failed infinite-total recipe.
