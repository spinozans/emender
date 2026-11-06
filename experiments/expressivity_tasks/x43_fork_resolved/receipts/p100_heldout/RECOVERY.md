# P100 exact continuation and recovery

Query verification is complete and pushed in bffaa99a. delivery.json records
verdict SHA and successful push; do not rerun or overwrite terminal verification.
All100000000000 raw bytes were scanned; full branch histogram checks passed.

The detached run_p100_heldout_leased.sh launcher waits for pushed verification,
then selects, extracts, and acquires broker card6 for frozen evaluation. Selection
is immutable in selection.json; windows are /home/erikg/c14_p100_heldout/windows.npz,
hash-bound there. All selected documents are after byte100e9; every selected
full-document digest was excluded against the entire hash-bound C14 offsets file.
No row-order identity assumption. Native extraction starts a separate read-only
serving process, leaving the completed union service untouched.

Inspect launch.json PID/starttime and descendant processes before recovery; never
duplicate a live extraction/evaluation. Selection is complete when
select.terminal.json exit0 and selection.json status passed. Feature extraction
writes extraction_progress.json after each16-window chunk, but has no supported
resume CLI: interrupted partial features.npy/contexts.npy are NOT a complete
cache. Preserve partial artifacts and progress receipt to a distinct retained
attempt directory first. If features.json is absent, fresh extraction may restart
from immutable windows with `python -m ...p100_heldout extract`, using ONLY the
approved interpreter and a new detached session with retained stdout/stderr.
Never overwrite a completed features.json cache. No extraction checkpoint or
unchanged bitwise resume is claimed.

If features.json is complete, hash-check both arrays and selection before fresh
broker card6 evaluation. p100_eval has no partial evaluation resume; preserve
partial eval_progress.json and logs, then run a new evaluation attempt from the
immutable cache, with fresh output destinations after explicitly retaining the
old outputs. No training, model calibration or weight update is involved. Do not
reuse the wrapper from its beginning after selection completed: selection refuses
overwrite, intentionally. Use the remaining phase directly in a detached broker
wrapper. When eval.json and terminal.json exit0 are complete, p100_report then
x43_result_delivery p100 harvest and push the final result; result delivery fails
closed if the Git index is already staged. Do not duplicate an active delivery
process. If push fails after a successful commit, push that exact commit and
retain evidence; do not create a replacement quality result.

Pure causal coverage is not identified: original C14 features used Wikipedia,
new features use Pile-head100, and duplicated substrings can remain indexed.
