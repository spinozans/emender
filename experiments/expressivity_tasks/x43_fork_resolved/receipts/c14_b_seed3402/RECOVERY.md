# Seed3402 exact recovery

Original detached wrapper automatically performs budgeted train, unbudgeted
train, budgeted eval, unbudgeted eval on broker card2. Do not duplicate it.
Each train saves every120 active seconds: model, optimizer, torch/CUDA/dropout
RNG, next_window, steps, real targets, telemetry and curve. Checkpoints outside
git: /home/erikg/c14_dense_bytes/emender_c14_b_seed3402_{budgeted,unbudgeted}.pt.
training.json binds the last complete atomic checkpoint hash and cursor. Console
steps after that receipt are uncheckpointed until the next save; on death report
that span honestly. Evaluation saves an atomic per-shard eval_resume.json.

First verify original launch PID/starttime is dead and its broker lease absent.
Compare the checkpoint SHA256 and training.json, and load optimizer/RNG/counters.
The new recovery wrapper skips completed arms, resumes each existing checkpoint,
and resumes evaluation only if that arm's eval_resume.json exists:
run_c14_b_seed3402_resume_leased.sh. Launch it with the required Popen detached
method (stdin DEVNULL, retained stdout/stderr, start_new_session=True), never as
a session-child/background shell. Broker owner and heartbeat live inside it.
Verify PPID1, matching owner/starttime, a renewal and advancing steps afterward.
No sign verdict until both complete35419 updates/71677042 targets and both full
frozen-window evaluations complete. Use66 rows and exact original denominators.
