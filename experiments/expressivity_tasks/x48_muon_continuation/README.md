# X48: continued training with hybrid Muon

**Complete:** [Direct retrieval beat optional learned execution](RESULT.md) on paired pre-update TRAIN loss and used less GPU time. Both arms reached the full dose; the holdouts remain closed.

[Protocol](PROTOCOL.md). A fresh update consumes 32 different complete 129-byte document windows, not a rate test on a repeated fixture. `train.py` updates the retained 150M GDN2 **and** its prediction head(s) with the same byte loss. On DIRECT/ENTER it also credits the selected memory trajectory. A byte-exact T4 frozen feature file is **not** used; the live host uses T1's BF16 chunk compute / FP32 master parameters, not FP32 serial serving. Do not equate the initial scores with S14. The direct-only arm never acquires a positional or private session worker. The program arm acquires permissions at admission and charges only actual launches.

**Prepared data:** `output/x48-muon/data/`, private. 32,768 full-document-verified TRAIN windows; 4,194,304 distinct targets per selected trajectory. Registered source manifest SHA `48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414`; ordered row metadata SHA `54c38b57716b8204c3aae8b4983b7a88a0fa76193752f9fc1af4dcb17d4b52bd`; window payload SHA `50cc533fcd60cf88d0061bc7a7b992b0cb96dbfe73dd1addb3d1ea9ee58db72c`; semantic identity `c40b0eacaa2af2f87ac14217854bc5399dc1dfd82b31b1cacb514af8e8fe1596`. No validation/test forward and no fitted model is in this directory. Missing data can be rebuilt from the registered local corpus by `python -m experiments.expressivity_tasks.x48_muon_continuation.data` with CUDA hidden; differing bytes, selected rows or source identities fail closed. Do not overwrite a partial extraction.

**How to run locally.** Use `/home/erikg/emender/.venv/bin/python`, root of repo on lambda01. The resource decision is independent of the X47 serving window. An exact TRAIN-only scope/arm and total cap/extension and frozen refs must be recorded in private `output/x48-muon/operator-authorization.json` **only upon explicit operator confirmation**; its absence intentionally blocks acquisition. Do not synthesize authorization to get past preflight. The source/gated refs bind the actual trained code, not `origin/main`.

```bash
P=/home/erikg/emender/.venv/bin/python
M=experiments.expressivity_tasks.x48_muon_continuation
# Once per frozen source (source-ref set before qualification), CUDA invisible:
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 "$P" -m "$M.qualification"
# After explicit cap AND time-window release only:
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 "$P" -m "$M.lease" check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 "$P" -m "$M.lease" launch
```

`launch` starts a detached CPU driver with retained `panel/console`. A short-lived intermediate parent detaches each broker-owning `leased.sh` (one GPU, direct then program, no automatic retries). Per-arm `<arm>/survival.jsonl` is the independent witness. Check PPid, owner starttime, advancing actual optimizer steps and renewal against the **current** broker file before reporting that a training remains active. `<arm>/{L,H,main}/steps.jsonl` contains every raw pre-update all-byte NLL, work, forward/backward/update wall. Every 16 steps measure actual per-parameter-group RMS displacement, not just matrix gradient norm. A parameter update is not by itself evidence that the program is useful. Account 1,152 physical updates/arm (both LR candidates, then carry the selected one from 128 through 1,024); comparisons after election use new windows, not the tuned-on prefix. Health checks are not free held-out evaluations.

**Stop/recovery:** retain all refusal/partial logs. `latest.json` atomically names the last fsynced private checkpoint, step, SHA and full binding. Main restores host/controller plus both hybrids (including their auxiliary AdamW states), not weights only. After any interruption first verify owner death, absence of its broker lease, and actual release/accounting. Compare `progress.json` / `steps.jsonl` with the last complete, SHA-verified step. Record any observed but unsaved span. No second lease or restart from zero is implicit; a partial requires a new bounded recovery decision. The initial paid entrypoint does not silently resume failed attempts.

**After both releases**, `python -m …x48_muon_continuation.compare` checks the selected checkpoint lineage, target counts and batch hashes. Its result is predictive loss on new TRAIN documents at the time they were consumed, not held-out quality, not an AdamW comparison, and not a test isolating the contribution of changed choices. No weights, raw window payloads or raw routes belong in Git.
