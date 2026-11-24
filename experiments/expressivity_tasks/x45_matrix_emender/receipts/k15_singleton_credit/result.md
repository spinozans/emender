# Stage O: qualified on train, did not pass development

On 48 held-out documents (12,288 targets), addressed outcome history improved all-byte NLL over the trained stateless and same-weights cleared readouts. But the trained **unkeyed** outcome history was better still. The preregistered gate required every comparison to gain at least 0.001 nat/byte with a positive lower paired 95% bound. It failed. No confirmation payloads opened; no confirmed address-specific benefit. N's separate batch16 veto stands.

| Control | Dev NLL (nats/byte) | Control minus writable (95% paired document bootstrap CI) |
|---|---:|---:|
| Trained unkeyed history | 1.174311 | −0.038607 [−0.050466, −0.027826] |
| Trained stateless | 1.226624 | +0.013706 [+0.007063, +0.021958] |
| Cleared writable | 1.228847 | +0.015929 [+0.008729, +0.024819] |
| Fixed online rule, fitted intercept | 1.256084 | +0.043166 [+0.032935, +0.054690] |
| Support formula, fitted intercept | 1.271120 | +0.058202 [+0.043070, +0.075824] |
| Fitted constant mixture | 1.278574 | +0.065656 [+0.050731, +0.083118] |

Writable NLL: 1.212918. All arms received the same singleton local probabilities and nonempty 8/4/2 backoff histograms. The base, SXGC lookup, delayed post-observation update, decay/shrinkage and two-byte addressing rule were fixed. Train-only Adam fitted the learned gates on identical 96 documents × 256 targets (128 steps each). Unkeyed pooled the same update into one document-local row; reset used writable weights with credit zeroed. An advantage over reset does not establish an advantage from addressing. The data do not distinguish why pooling helped. Bootstrap intervals describe these dev documents; no confirmation estimate.

Independent singleton dense/sequential checks on 32 positions of the first train and first dev document passed for all seven arms, worst max abs 5.56e-17 and 2.23e-16 respectively (rounded up). These are prefix checks, not exhaustive deployment equivalence. Eight-document extraction took 115.693s (~56.49ms/target), pricing 2967.60s if both panels were used. Runtime CPU verification checked 144 document IDs/steps/shapes/cache SHA256s and recomputed NLLs, paired CIs, gates, parity thresholds, checkpoint-panel binding and zero locked caches.

GPU ledger: lambda01 card 0, detached owner PID/SID 3844285, PPID 1, proc starttime 551685020. Creation epoch 1795450862, last recorded heartbeat 1795452885, released 1795452887. Integer timestamps differ by 2025s; allowing subsecond rounding, O charged <=2026 card-seconds (0.563 h). Prior <=4402s; cumulative <=6428s (1.786 h) versus 21600s cap. Lease file absent, owner gone at CPU closeout. No restart.

Paths relative to `/home/erikg/emender`: caches and runtime files `output/k15O/`; audited receipts in `experiments/expressivity_tasks/x45_matrix_emender/receipts/k15_singleton_credit/` (including `checkpoint.json`, `verification.json`, `console`, `heartbeat`, `initial`, `live.txt`, `launch.json`, `released`, `dev.json`, `dev-parity.json`). Protocol `experiments/expressivity_tasks/x45_matrix_emender/K15_SINGLETON_CREDIT.md` committed b3f4d5e5; executable source/tests 45122871. Replay CLI in `k15_singleton_credit.py --replay PREFIX NPY` (module invocation under project venv with approved lease; not relaunched). CPU check: `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.k15_singleton_verify`.
