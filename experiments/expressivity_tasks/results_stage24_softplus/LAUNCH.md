# X24 launch receipt

Chosen activation: plain softplus; Concession #18 and fused-engine amendment
are registered. Preregistration commit **fca01ad8fc2d119cd50a27ea580b7275b4fcad41**
was pushed before launch; launch.json retains the matching remote ref.
Registration SHA256: 68b295cf18aad0c74833734e394e252950a427ce06162159235359c3257e692a.

Launch: receipts/launch.json. Card7, host lambda01; watcher3126804,
worker3127171 (own process group), trainer3127273. PID starttime identities,
physical GPU snapshot and observed fused `state=softplus eager_fallback=False`
are retained in receipts/launch_verified.json. Config binds source commit,
registration hash and the exact shared X23E graph dataset receipt hashes.

State at verification: running, initial baseline validation pending; no scientific
outcome or length-generalization result yet. All three PID identities alive.
Persistent60s watcher enforces disk/wall bounds; --once observed running (exit2).
Launch command: PYTHON_BIN=$PWD/.venv/bin/python OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nohup $PWD/.venv/bin/python -u -m scripts.watch_e97_softplus_graph --launch.

Native supervisor arming is wired through receipts/watcher_wiring.json and
reported to the supervisor; no native notification activation is claimed here.
Exact probe: .venv/bin/python -m scripts.watch_e97_softplus_graph --once.
Terminal target: receipts/watcher_terminal.json (absent while running).

Final prelaunch kernel gate worst max-relative errors across production/direct
and projection-chunk cases: output .00543220, state .00492043, input-gradient
.00739461, parameter-gradient .01076973, evaluation .00554431. Tolerances:
output/state/eval .03, gradients .08. 87 regression tests passed; max-bin
90,904-token zero-update smoke passed (peak10,290,562,560 bytes).

Review: independent acceptance review required; no experiment outcome adjudicated.
Mutable training logs/checkpoints and watcher polls are intentionally not staged.
