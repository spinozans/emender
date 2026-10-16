# X1 report — revised Stage 1, invertible nonlinear-map benchmark

## Result and scope

Implemented the operator's two-coordinate update `(x,y) -> (y,x+y²+c) mod p`
with its exact inverse, full-state targets, full-field drivers, and initial
state supplied once. **No GPU work, no training, no Slurm submission, and no
Stage-2 launch.** Recommend production **p=61** (3,721 full states), following
small-prime p=13/31 pilots; this is a practical recommendation, not measured
learnability. Source and framing are in
`experiments/expressivity_tasks/INVERTIBLE_QUADRATIC.md`.

## Changed files

- `experiments/expressivity_tasks/tasks/invertible_quadratic.py`: forward,
  inverse, trajectory, full-state task, one-step variant, suffix-only MAP
  baseline, exact old mod-256 parity+seven-driver baseline and its labeled
  negative-control transfer.
- `experiments/expressivity_tasks/tasks/__init__.py`: six registered task
  variants (production and both pilots, plus each one's one-step diagnostic).
- `tests/test_invertible_quadratic.py`: 43 collected CPU test cases.
- `experiments/expressivity_tasks/verify_invertible_quadratic.py`: reproducible
  CPU empirical caps and data/oracle pilots, fail-closed statistical checks.
- `experiments/expressivity_tasks/invertible_quadratic_cpu.json`: retained
  seeded measurements, counts, sample sizes, deviations, and pilot results.
- `experiments/expressivity_tasks/INVERTIBLE_QUADRATIC.md`: formal proof,
  encoding, old calibration proof, generated empirical table, p recommendation,
  length schedule, and framing limits.
- `experiments/expressivity_tasks/INVERTIBLE_QUADRATIC_STAGE2_PLAN.md`: designed,
  explicitly unlaunched Stage-2 arms and checkpoint/reporting/approval gates.
- `X1-report.md`: this report.

## Mandatory checklist evidence

1. Exact inverse: exhaustive all states/all drivers for p=3/5/7, both round-trip
   directions; 10,000 random cases each for p=13/31/61.
2. No collision: exhaustive fixed-driver bijections for p=3/5/7, shared
   length-128 sequences; randomized distinct-state production checks for 100
   drivers plus a shared length-128 production sequence.
3. Old baseline: every position of 4,096 old length-128 trajectories matches
   exactly, with randomized old initialization and an explicit original x0=1
   check. New-map exhaustive hidden-two-driver controls show p² distinct true
   states but <=2 old-summary guesses. Empirical transfer measurements below
   are at chance, not exact-reproduction success.
4. Task streams: repeated RNG seeds match across successive variable-size
   batches. Initial coordinates and full-field drivers each pass predeclared
   six-sigma marginal uniformity tests on 200,000 samples per field.
5. One-step: task requires exactly three tokens, and independent coordinate
   oracles validate 10,000 labels each at p=13/31/61. This establishes diagnostic
   correctness, **not that any model has learned the arithmetic**.
6. Input flow: independently reconstructed RNG draws verify two typed initial
   coordinates once, only drivers thereafter, no target feedback, prefix-only
   masking, and exact joint-state labels checked by a separate scalar oracle.
7. Suffix lemma: formal statement/proof includes its t>=k+2 boundary and
   conditioning on the known initial state. Exhaustive tests condition on
   every small-field initial state, enumerate two hidden drivers, and verify
   exactly one correct full-state guess out of p² for suffix lengths 0/1/7.
8. CPU data pilots at 128/512/1024 drivers (130/514/1026 total tokens) all
   reproduce exact oracle targets at p=13/31/61. Production p=61 keeps the
   joint head practical (about 3.81M weights at dim1024), while expanding the
   old 256-state space by >14x. Learned pilots await approval.

## Empirical cap measurements (accuracy fractions, not percentages)

Each p has **2,000,000 independent terminal observations**, fixed **known**
initial state (1,1), 15 iid drivers, seed 20261016+p. Suffix seven hides eight
prefix drivers. The fixed six-sigma check passes for every predictor. A finite
sample can lie above the exact population cap; that does not violate the proof.

| p | States | Exact 1/p² | Suffix 0 | Suffix 1 | Suffix 7 | Old parity transfer | Max abs z |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 169 | 0.0059171598 | 0.0059225 | 0.0058780 | 0.0060020 | 0.0059405 | 1.564 |
| 31 | 961 | 0.0010405827 | 0.0010730 | 0.0010340 | 0.0010190 | 0.0009750 | 2.877 |
| 61 | 3,721 | 0.0002687450 | 0.0002650 | 0.0002735 | 0.0002815 | 0.0002955 | 2.308 |

Artifact SHA256:
`31ccc326060f7c1f02ba092c63f655fb26e974f3d5692c366e4abf81dd0cdf4f`.

Important distinction: the old parity+suffix method remembers an extra prefix
bit, so it is NOT a suffix-only observer and is not covered by the 1/p² lemma.
Its new-map transfer is an explicitly empirical negative control, not a
universal theorem about all compressed prefix summaries. The package docs
state this boundary, including the more general 2/p² counting bound for a
one-bit prefix summary.

## Commands and results

Host: **lambda01**, local `/usr/bin/python3` 3.12.3, NumPy 1.26.4,
pytest 9.0.2. GPU visibility was explicitly empty for all Python validation.
The required Frontier activation was attempted first and refused with status69
because this is not Frontier and `module` is absent. No guessed Frontier
interpreter or replacement module stack was used.

```text
source scripts/frontier/activate_emender_frontier.sh
  -> expected environment refusal: module unavailable (status 69); host is lambda01
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m pytest -q tests/test_invertible_quadratic.py
  -> 43 passed in 1.56s
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m experiments.expressivity_tasks.verify_invertible_quadratic --output experiments/expressivity_tasks/invertible_quadratic_cpu.json
  -> all 12 empirical checks passed; all nine depth pilots and three one-step oracle panels passed
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m experiments.expressivity_tasks.verify_invertible_quadratic --output /tmp/X1-reproduced.json
cmp experiments/expressivity_tasks/invertible_quadratic_cpu.json /tmp/X1-reproduced.json
  -> independent rerun reproduced retained JSON byte-for-byte; all documented table values match JSON
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m pytest --collect-only -q tests/test_invertible_quadratic.py
  -> 43 collected, successful
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m pytest --collect-only -q tests -m 'not gpu and not cuda'
  -> 2258/2333 tests collected (75 deselected) in 13.94s
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m compileall -q experiments/expressivity_tasks/tasks/invertible_quadratic.py experiments/expressivity_tasks/verify_invertible_quadratic.py tests/test_invertible_quadratic.py
  -> passed
cmp AGENTS.md CLAUDE.md
  -> identical; neither changed
git diff --check
  -> passed
```

The repository-wide CPU suite was **collected, not executed**. The task suite
was executed fully. Test counts above were derived from pytest output rather
than edited from an expected total. CI discovers `tests/test_*.py` and has no
hardcoded pytest-count preflight. The bank launcher's numeric preflight is for
its separate bank self-test suite, which this change does not modify; no bank
count literal was changed. No resilient DiLoCo behavior is modified or tested,
so its architecture/gap-matrix requirement IDs do not apply to this task.

## Stage-2 plan and constraints

The unlaunched plan specifies matched **18,269 actual optimizer updates** for
E97 and GDN2 (also the same-geometry E97 identity-state ablation), best-validation
checkpoint selection for **both** main arms, separate held-out test pools,
matched-step panels alongside terminal and best-validation panels, suffix and
one-step calibrations, paired seeds and explicit parameter accounting.

Training depth follows the old 128-driver protocol, with primary extrapolation
128 -> 512 -> 1024 drivers (x1/x4/x8). The two initialization tokens are added
once at every length. The old trainer must first be adapted to the documented
checkpoint/validation contract after approval; it is not silently treated as
already implementing it. No run commands were submitted.

This is an efficiency/learned-computation benchmark at practical state sizes.
Every finite-state task admits a sufficiently large one-hot linear-state
representation. **No impossibility or universal expressivity claim, and no
CMA-causal account of GDN2's earlier failure, is made.** Invertibility prevents
trajectory merging but does not prove learnability or rule out every shortcut.

## Residual risks and next step

- Learned arithmetic, optimization behavior, and length extrapolation are
  unmeasured by design; one-step learned failures would block depth claims.
- Empirical six-sigma checks are calibration, not proof; the proof's exact cap
  applies only under its explicitly stated observer/driver conditions.
- Full repository test execution and independent acceptance review remain
  outside this focused CPU validation. No known focused-test blocker remains.
- Next: independent review, then explicit operator approval before any Stage-2
  runner changes/training/GPU work. The commit containing this report is pushed
  as the scoped X1 deliverable; final git status is checked after commit/push.
