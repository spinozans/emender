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
