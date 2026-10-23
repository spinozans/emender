# X26 launch receipt

Registration **af649fd8** pushed to origin/main BEFORE launch; full commit/remote
ref retained in receipts/launch.json. Independent launch_verified.json checks
source/dependency/data receipts, actual trainer configs (batch128; constant
AdamW lr1e-4/wd1e-4), worker+trainer PID/starttime/host, owned process groups,
and trainer PID-to-physical GPU UUID via nvidia-smi compute-app output.

lambda01: RNNgraph card2 worker501864 trainer502181;
Deltagraph card3 worker501865 trainer502195;
RNNinteger card4 worker501867 trainer502209;
Deltainteger card5 worker501869 trainer502223.
Persistent independent60s resource watcher501784. All four running at verification.
Long RNN unary unroll may be slow; no completion-time/scientific-success claim.
Other experiments/cards0,1,7 untouched. Card6 released after zero-update smokes.
RWKV optional arms NOT launched: upstream v_first=None failure, raw receipts.
Repo-default secondary NOT launched: upstream shell/parser inconsistency.

Standard chain: `PYTHON_BIN=$PWD/.venv/bin/python OMP_NUM_THREADS=2
MKL_NUM_THREADS=2 nohup .venv/bin/python -u -m scripts.watch_merrill26_replication
--launch`. Fresh models and shared actual X23E datasets, never restored weights.
Per-rung40GiB/7day and400GiB minimum-free guard; identity-checked TERM then KILL.
Watcher wiring: receipts/watcher_wiring.json; native notification arming requested
through coordination channel, supervisor-owned and not claimed armed by worker.
Probe `.venv/bin/python -m scripts.watch_merrill26_replication --once`: exit2,
all four running; target receipts/watcher_terminal.json absent while active.

32 tests and four max-bin zero-update smokes pass; optional RWKV probes fail.
Exact Table1 receipts and approved departures in DISCREPANCIES.md. This remains
**partial replication under reconciled sources**, not exact Figure2 reproduction.
Independent acceptance review required. No outcomes adjudicated at launch.
Mutable logs/checkpoints/status/polls and unrelated dirty files are not staged.
