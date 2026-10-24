# X26b validation report

Local lambda01 `.venv/bin/python`3.12.3/CUDA; not Frontier. No resilient training
changes; R01–R16/NDP01–NDP17/V21S01–V21S17/ISP01–ISP07 not applicable.
No Slurm queue binding or WorksGood service invoked.

- `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q
  tests/test_merrill26_native_control.py tests/test_merrill26_replication.py
  tests/test_merrill26_separators.py tests/test_recipe_controlled_separators.py`:
  **38 passed**, pre-registration8.07s, postlaunch8.15s. Raw prereg receipts/tests.txt;
  postlaunch `/tmp/X26b-final-tests.txt`. Six native checks cover exact source
  models/encoder dimensions, BOS/MAT layout and no STATE leakage, ==0 vs>0 prefix
  labels, unchanged native BCE functions, identical GRU/Delta collates, fixed
  cards/recipe/curriculum. Original task/generator checks still pass.
- Shell `bash -n scripts/launch_merrill26_native_control.sh`: pass; compileall
  runner/register/watcher: pass. Source snapshots independently checked against
  actual upstream git blobs at dbfa365a before registration, hashes unchanged.
- Native generator unmodified, seed0 and approved per-length counts; ten native
  dataset files hash-bound via deterministic command + receipts/data.json.
  Uniform thirds vs X23E(.45,.10,.45), exact vs int64, balancing, label and
  tokenization differences explicitly compared. No OOD performance inspected
  or used for training/selection; label distributions reported as baselines.
- Real CUDA zero-update smokes on leased physical4/5: GRU793857parameters,
  batch128/N300/301tokens,2.62s,508469760B peak; Delta270401parameters,28.79s,
  669923328B peak (includes compilation). Backward, finite gradients and eval pass.
- Registration7251c265 pushed before launch; persistent guard and two GPU trainer
  identities/configs/UUIDs verified, raw receipt launch_verified.json.
- `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m
  scripts.watch_merrill26_native_control --once`: expected exit2, both running.
- Graph optimized candidate NOT qualified: two tests pass/one fails, raw
  scan_tests.txt. Ordinary short model cases pass but saturation BF16 input-grad
  error.2734375 exceeds.08. Source archived outside production; no graph relaunch.
  Eager resource-censor receipt binds measured steps/rate/CPU/GPU and terminal.

Self-review: no known native-control blockers. Native data/protocol bundle is not
single-variable; only bundle-level localization possible. Native control outcomes
remain pending. Graph optimization remains explicitly blocked on equivalence.
Independent acceptance review still required. No mutable logs/checkpoints/datasets
or unrelated runtime modifications staged; only named implementation/static receipts.
