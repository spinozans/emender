# E97 dose-screen harness (v1) — STAGED, NOT EXECUTED

Repo mirror of the staged dose-screen machinery. The LIVE harness root (with
build logs, receipts, and DRAFT validation artifacts) is
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-dose-screen-v1/`.
Authoritative report: `docs/validation/e97-dose-screen-harness-v1.md`.

Contents (CPU-staged; the GPU probes are executed by the parent orchestrator
with operator sign-off at admission):

- `build_prep.sh` / `build_dose.sh` — the prep/packs/validation/audit build
  scripts used to produce `pi-native-repair9-full-preparation-v3-d{25,50,75}`
  (v3 byte-reproduction validated first: prep `3c4f0b38...`, packs
  `5a4baffb...`).
- `freeze_dose_probe_proposal.py` — fail-closed 32-update probe-proposal
  writer; the three DRAFT freezes in `proposals/` PASS the generic auditor
  (receipts in `proposal-audits/`).
- `run-probe.sh` + `d{25,50,75}/run-probe.sh` — probe-dir instantiation
  (worktree at d77dcc46, run-train.sh [steps 32, save-every 32, chunk
  2048/16384, lr 1e-5, bridge parent], run-stageb.sh, execution-slice
  runners with per-checkpoint panel binding guards).
- `d{25,50,75}/commands.sh` — the exact orchestrator command sheets
  (freeze -> audit -> commit -> ADMISSION -> instantiate -> train -> Stage-B
  -> slice -> readings).
- `execution-slice/` — the fixed seeded 24-case execution-slice definition
  (seed 240977; 6 id-prefix groups x 4 families), the per-checkpoint slice
  panel freezer, and the selection documentation.
- `read-dose-results.py` — post-probe readings collector.
- `dose-identities.json` — the frozen identity table (prep/packs/schedule/
  proposal/audit sha256 per dose).

Nothing here has been admitted, trained, evaluated, or promoted; no threshold
changed; scoped commits only.
