# X42 three-family velocity search

The live pure-emender CMA-ES instance is unchanged. Two additional instances
search the transformer (6 dimensions) and external GDN2 (9 dimensions) adapters
from `velocity_baselines.py`. M2RNN is related work only.

Each search uses population 12, an absolute 12-generation cap, and an independent
atomic generation state containing population, RNG, covariance, paths, phase,
and completed scores. A failed child stops that family without scoring it;
restart retries that candidate from scratch and preserves completed candidates.
Each family has one outstanding candidate, so baseline searches can occupy at
most two cards. No candidate preempts an existing lease.

The registered clock is 2700 synchronized seconds for data loading and complete
B64/T2048 updates. Microbatch is a divisor of 64; weighted backward accumulation
finishes before the one clipping/AdamW update. Evaluation uses the same first 24
T2048 held-out windows. Baseline score is held-out NLL only. Baseline winners and
the pure winner must ultimately be compared by raw NLL, acknowledging the pure
search's occupancy and degeneracy objective terms.

The transformer searches learning rate, warmup, weight decay, microbatch, weight
initialization scale, and AdamW beta2. Its anchor architecture stays fixed. GDN2
also searches initial decay rate scale, initial dt-bias offset, and expansion
(1 or 2); the existing capacity solver rematches width/depth/MLP for expansion.
Both remain within 0.1% of 150,003,984 parameters. The registered external wrapper
does not expose kernel chunk length, so the native chunk path stays fixed.

The committed `velocity_transformer_registration.json` and
`velocity_gdn2_registration.json` bind dimensions, score, source hashes, external
GDN2 Python sources, FLA 0.5.1, corpus authority, and the production regime before
any scored candidate. The anchor scripts and original search sources are byte
unchanged, preserving their source checks. Generation-zero anchor results are
references only and are imported when they appear; they never enter CMA tell.

## Managed instances

| Family | User systemd unit | Run directory under `runs/` |
| --- | --- | --- |
| Pure emender | `emender-velocity-loop` | `velocity_production_v1` |
| Transformer | `emender-velocity-transformer` | `velocity_transformer_v1` |
| GDN2 | `emender-velocity-gdn2` | `velocity_gdn2_v1` |

The baseline units use the approved `/home/erikg/emender/.venv/bin/python`,
`velocity_baseline_parent`, their respective `--arm` and `--registration`, the
run directory as `--output`, and `--generations 12`. They are transient units,
matching the existing session services. Use `systemctl --user restart UNIT`
to resume after a child failure. A reboot requires recreating the transient
units with the same arguments; generation state survives on disk.

The wrapper uses the existing atomic lease broker on cards 1/5/6. It polls less
frequently (11 seconds) than the pure wrapper (5 seconds) to favor the pure
search. This is a polling preference, not a strict scheduler priority or fairness
guarantee. Baselines may remain queued while the pure pool is saturated. Card 0
is considered only after both depth and anchor services are inactive and both
anchor result files exist. Cards 2/3/4/7 are never eligible. The WorksGood
service, dispatcher, workers, and completion machinery are not used.

`launchN/queued.json` receipts attest dispatch, `launchN/lease.txt` attests actual
acquisition, `attemptN/start.json` attests model setup, and `result.json` attests
a completed score. A queued launch is never represented as a trained result.

## Validation

The focused suite covers both families' capacity and space boundaries, mock
child failures, partial resume without rescoring, generation phase and cap,
endpoint rejection, and a real CPU optimizer equivalence check for accumulated
B8 versus B64. Existing anchor AST-loop/capacity tests and pure resume/state tests
also pass (16 tests). The first boundary test exposed floating-point endpoints;
decode now clamps to the registered bounds. Initial service preflight failed on
an inherited verifier's absolute Git path; the new caller now supplies a
repository-relative path. Both initial failures are retained as
`startup_failure_repository_path.json` in the run directories. Neither produced
a scored candidate.

No resilient DiLoCo behavior is changed; compute-pool gap-matrix requirement IDs
are not applicable to this single-device search orchestration.
