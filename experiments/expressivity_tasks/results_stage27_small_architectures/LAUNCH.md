# X27 launch receipt

Pre-training registration commit **d3cac70bf44731686fb47f583c8bab3176da6ded** was
pushed to origin/main before launch; `receipts/launch.json` retains the exact
matching `git ls-remote` result and physical-GPU UUID snapshot. No X27 optimizer
updates preceded that registration. Fresh max-N300/90904-token backward+eval
smokes passed for all three arms with zero optimizer updates.

- E97 tanh: depth1/dim192/heads16/state16,570464 params, launched on physical6.
- E97 softplus: same size/count, queued for idle/unleased physical0 after X25.
- GDN2: depth1/dim128/expand8.625, effective heads1/key128/value1104,
  563218 params, queued for idle/unleased physical1 after X25.

Actual training source commit and parameter count are in
`e97_graph_seed0/config.json`; worker/trainer identity, exact command and lease
are in `receipts/e97_graph_seed0/`. `receipts/postlaunch_verification.json`
confirms registered source/dependency hashes, pushed registration identity,
source commit/config agreement, GPU6 binding, shared actual dataset link and
live worker/trainer/independent guard identities. Baseline validation is not
an optimizer update or a scientific outcome.

Persistent watcher PID2037421/starttime285167293 on lambda01 polls every60s;
`receipts/watcher_wiring.json` gives the precise paths and supervisor probe:

```
.venv/bin/python -m scripts.watch_small_architecture_separators --once
```

Exit2 means still nonterminal (running or queued), not a failed check. Native
notification arming is supervisor-owned; this launch only claims its independent
resource guard and exposed wiring. Terminals require result/marker SHA agreement;
resource interruptions are right-censored, not architecture verdicts.

Validation:50 CPU tests pass both pre- and post-launch; shell syntax passes;
all three max-length CUDA gates pass. Scoped staged diff has no whitespace
errors. Unrelated pre-existing/live watcher modifications and outputs were not
staged or changed by this task. R01–R16/NDP01–NDP17/V21S01–V21S17/ISP01–ISP07
are not applicable (local nonresilient research; no Frontier Slurm claim).
Independent acceptance review remains required; no convergence claim.

## X27q queue recovery — 2026-10-24 UTC

Fix/amendment **50f9fe8dbe0ae45fb08ce7a7c41c4e0aff6b77e9** was pushed before
restarting `bash scripts/launch_small_architecture_separators.sh` (`--launch`).
Persistent launcher PID1384720/starttime291042314 on lambda01 now polls every60s.
Softplus-small launched on preferred physical0, worker1385276/trainer1385490.
GDN2-small launched on next-free physical2 (preferred1 busy),
worker1385615/trainer1385724. Existing tanh worker was not relaunched.

Each new arm has `receipts/<arm>/launch.json`, worker/trainer identity, lease
and command receipts. `receipts/queuefix/postlaunch_verification.json` verifies
pushed HEAD, all236 source/389 dependency hashes, matching config source commit,
live processes, actual nvidia-smi GPU UUID/PID bindings, and unchanged original
shared launch receipt. The existing native `--once` probe returns2, all running;
no new notification arming was attempted. Supervisor-side native-wrapper
verification requested separately; no scientific result is claimed.

Parser tests cover single/multiple comma-separated cards, empty and whitespace,
trailing newline and mixed separators. Restart regression exercises real per-arm
receipt creation and shared launch gate: A unchanged, B launched once, duplicate
B rejected. Default one-shot and non-X27 rejection tests cover scope isolation.
24 X27 tests /62 five-family regression tests pass before and after launch.
Shell syntax and code/amendment staged whitespace checks pass. Generated command
receipts retain the wrapper's harmless trailing space verbatim. Global whitespace check
found only unrelated live stage14/stage16 report EOF whitespace, left untouched.
Other experiment registrations were not rewritten; a future relaunch of another
family using the changed shared watcher must qualify its own source bindings.
Partial spawn failures remain fail-closed. Independent acceptance review required.
