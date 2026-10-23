# X25 launch receipt

Append-only recipe amendment commit **36bd9d91** pushed to origin/main BEFORE
launch. `receipts/launch.json` retains full commit and matching remote ref.
`receipts/launch_verified.json` independently verifies all five live worker/trainer
PID/starttime/host identities, owned groups, physical GPU UUID binding, actual
lr/wd1e-4 optimizer configs, shared dataset hashes and committed source bytes.

Host lambda01; persistent60s resource watcher PID4175328.
- card0 e97_graph_seed0, tanh: worker4175674, trainer4175977.
- card7 e97softplus_graph_seed0, plain softplus: worker4175675, trainer4175994.
- card1 gdn2_graph_seed0: worker4175677, trainer4176006.
- card2 e97_integer_seed0, tanh: worker4175680, trainer4176035.
- card3 gdn2_integer_seed0: worker4175682, trainer4176019.

Observed E97 logs explicitly name correct tanh/softplus activation and
`eager_fallback=False`. New configs bind amendment commit/registration and
paper optimizer. Fresh namespace; no restored weights or hot-recipe checkpoints.
Free disk at launch1469107122176 bytes, above400GiB. Per-rung40GiB,7day wall
and independent60s identity-checked TERM/KILL guards active. No scientific verdict:
baseline validation/training underway; all outcomes remain pending.

Hot controls remain at their original X23E/X24 paths. Both
`receipts/e97_graph_seed0/recipe_superseded.json` files bind preserved checkpoints
and matching terminal hashes; result+terminal+CENSORED marker label
RECIPE-CONFOUNDED/SUPERSEDED, explicitly not training verdicts. GDN2 diagnosis:
receipts/gdn2_nonfinite_diagnosis.json, existing traceback only, no state telemetry.

Native supervisor arming wiring: receipts/watcher_wiring.json.
Probe `.venv/bin/python -m scripts.watch_recipe_controlled_separators --once`
returned2, all five running. Terminal target receipts/watcher_terminal.json,
absent while active; status receipts/watcher_status.json. Supervisor notified via
coordination channel; this worker does not claim native notification arming.

Launch command: `PYTHON_BIN=$PWD/.venv/bin/python OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
 nohup $PWD/.venv/bin/python -u -m scripts.watch_recipe_controlled_separators --launch`.

Validation:28 tests passed, shell syntax and scoped diff check passed. Historical
zero-update max-bin and X24 kernel gates reused with unchanged model/kernel bytes.
No new instrumented GDN2 runs. No mod97/control relaunch. No resilient DiLoCo or
Frontier queue binding claimed. Independent acceptance review remains required.
Mutable logs/checkpoints/polls and unrelated watcher edits are not staged.
