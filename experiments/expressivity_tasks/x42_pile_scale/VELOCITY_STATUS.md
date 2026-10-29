# X42 velocity launch

Preregistration committed as `e0ec61ac`, SHA256
`f394fc0680f86ac806976a3162d321561087e92c0837da0677303c6a8c51a131`,
before the first scored candidate. The direct detached parent PID 974815 owns
`runs/velocity_v1`. Smoke is running on broker-leased card 1, targeting 2700
active training seconds. Its first full update consumed 512 tokens in 11.15
seconds. This is a live launch receipt, not a completed smoke score or generation.

After a successful smoke score, the same parent starts population 12, with up
to three direct child processes on broker cards 1/5/6. Generation means,
covariance, population, evolution paths, RNG and completed scores are durable.
Restart by running the same `velocity_parent` command and output directory;
verified surviving child groups are stopped and incomplete candidates restart
from scratch. No model checkpoint or .pt file is used.

Thirty local velocity tests passed, including SIGKILL generation-state resume.
The new isolated policy kernel passed exact state/trace/output/gradient checks
against unchanged X40 and independently reconstructed own-head occupancy.
Full 150M-trainable proxy train/eval integration passed for N1/3/4/5, vocabulary
50257, checkpoint/replay/backward/AdamW at T512 and telemetry at T2048.
See the hash-bound registration and compact integration/parity receipts.

The broader historical suite had 52 passes and two failures: old part8
checkpoint preflight receipts have stale backward.py hashes at T512/T2048.
Those failures remain failures. No certified source or parity gate was edited.
The current part12 support gate matches current core sources; its historical
parity harness hash itself is stale, although forward/backward core hashes match.

Residual limits: the 2700-second endpoint and generation 1 are pending; batch1
is not throughput tuned; gate-cost only offsets the frozen-management objective
and does not affect trainable gradients; proxy-to-scale transfer is unproven.
Card 0 stays excluded throughout. PID 634691 was checked alive and untouched;
2/3/4 and 7 are excluded. No WorksGood, Slurm or DiLoCo behavior was used.
Intercom/contact_supervisor tools were not exposed in this session.
