# X26c shipped RAW-DEFAULTS launches

Registration/code/gates commit **832c2115** pushed origin/main BEFORE launch.
`receipts/launch.json` binds the full commit, remote ref and registration hash.
Both full upstream bash launchers executed unchanged; their own console logs
include pip installs, environment summaries, literal training commands and
native training diagnostics. No unmodified upstream raw diagnostic should be
interpreted without the pending external balanced/recall report.

- GRU `rnn_integer_seed0`: physical card4 / trainer1693325,
  UUID GPU-76581993-9fdb-55b9-05b2-97621aac8629.
- DeltaNet `delta_integer_seed0`: physical card5 / trainer1693643,
  UUID GPU-ed7c9f77-649d-1583-1fd9-a4e8b160d94f.

Both have reached optimizer updates in their SHIPPED trainers. No training
outcome adjudicated; do not read initial updates as success/failure evidence.
`receipts/launch_verified.json` independently checks live worker/trainer
PID+starttime+host, physical UUID mapping, actual trainer argv/defaults,
no STATE teacher forcing, environment versions and selected visible card.
Runtime numpy2.5.4; Delta FLA0.6.0; reused torch2.9.1+cu128. Environment adaptation
and supervisor CE→shipped BCE correction are in ENVIRONMENT.md.

Watcher PID1692488 verified live; persistent60s identity-safe resource guard
(400GiB free, 40GiB/rung, 7days, TERM→KILL). At verification free1.445TB.
`receipts/watcher_wiring.json` retains watcher identity, probe, status and
terminal paths. Probe `.venv/bin/python -m scripts.watch_merrill26_raw_defaults
--once` returns2 with both arms running; this is EXPECTED nonterminal status.
Native notification arming remains supervisor-owned, not claimed here.

After a shipped pipeline exits successfully, the same worker runs the separate
frozen-best-ID-checkpoint evaluator in its isolated venv. `result.json` then
contains pooled balanced accuracy, both recalls, confusion matrices/class
counts/majority baselines, per-length metrics and immutable checkpoint SHA256
for validation and all three caption-matched test bins. A pipeline error or
resource censor produces no scientific verdict. Three registered outcomes
remain frozen; independent review and terminal metrics are still pending.

Postlaunch regression: **46 passed in7.71s**, receipts/tests_postlaunch.txt.
No edits to registered source or registration after launch; no other arms touched.
