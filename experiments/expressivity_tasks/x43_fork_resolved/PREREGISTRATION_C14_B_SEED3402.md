# C14-B second-seed matched-exposure amendment

Registered before measurement. Seed3402 replaces3401, for BOTH fresh budgeted
32/64 and unbudgeted257 arms, otherwise PREREGISTRATION_C14_B_MATCHED.md applies:
35419 updates /71677042 real targets each, B16/T128, frozen raw document windows,
masked padding, independent empty resets, AdamW3e-4/.01/clip1, BF16, keep.5.
This is a matched-exposure amendment to budgeted stopping as well as a seed
amendment; no historical endpoint is relabeled. Each arm has its own checkpoint.

Frozen c190956d sources are checked. Unbudgeted installs phase2 rowkeys and
reachable-tile wrapper after validating every phase2 source hash. Equivalence
citations and fingerprint hashes are inherited explicitly from
receipts/c14_b_matched/EVAL_LAUNCH.json (phase1 state/replay/gradient, phase2
rowkeys state/replay/gradient, phase3 repeated pairs and sustained training).
Budgeted retains its frozen historical kernel: the unbounded-only equivalence
receipt does not silently qualify a budgeted kernel substitution.

PT1 is new-seed unbudgeted minus new-seed budgeted final-test NLL. Negative
holds prior sign, positive flips. Magnitudes below.005 nats/byte will be called
practically near-zero (a descriptive threshold, not a statistical noise test).
Report actual difference and both seeds, no significance claim. Predict negative
but small, roughly0 to-.04 nats. All outcomes results. Frozen paired evaluation
includes all66 on/off split/stratum rows and original denominators. No verdict
before both full evaluations finish. Protocol remains limited to two seeds.

One broker lease on card2 runs both arms and their eval sequentially, alongside
E2 on4 and E3 CPU (eventual eval6). Detached session with internal broker heartbeat,
retained logs, atomic120-active-second checkpoints and exact --resume cursor,
optimizer and RNG state. No frozen arms or union service change. No DiLoCo work;
requirement IDs inapplicable. Supervisor transport unavailable in enabled tools.
