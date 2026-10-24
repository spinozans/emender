# X26c validation (before launch)

Lambda01 Python3.12.3/CUDA, not Frontier; no resilient DiLoCo changes.
R01–R16/NDP01–NDP17/V21S01–V21S17/ISP01–ISP07 not applicable.

- `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_merrill26_raw_defaults.py tests/test_merrill26_native_control.py tests/test_merrill26_replication.py tests/test_merrill26_separators.py tests/test_recipe_controlled_separators.py`: **46 passed** (8 new +38 existing).
- New tests: prior-tracking90% raw becomes50% balanced with recalls0/1;
  absent-class balanced=null; example-weighted aggregate; strict model/checkpoint
  round-trip both arms/no optimizer; immutable14-file snapshot hashes; shell
  cannot override training defaults; final-MAT-only per-length metrics.
- Initial two test invocations named nonexistent recipe-test paths: collected0,
  failed code4. Both error receipts retained; corrected exact command above green.
- Real max-bin N300/batch256 BF16 backward+finite-grad+eval smokes, zero updates:
  GRU peak997515776bytes; Delta1228252160bytes. Both isolated environments pass.
- External evaluator separately smoke-tested in each isolated venv with actual
  native first/last rows across all splits (matrix lengths1/100/101/200/201/300),
  temporary random frozen checkpoint, zero optimizer updates; strict reload,
  pooled counts/class recalls/per-length metrics and unchanged SHA256 pass.
- `bash -n scripts/launch_merrill26_raw_defaults.sh`: pass; Python compileall
  raw evaluator/register/watcher: pass. Registration checks all14 vendored file
  hashes against actual upstream git blobs, all10 frozen X26b data hashes,
  all4 completed trigger result/terminal hashes, both max-bin smokes.

No X26c training outcomes observed before registration. Mutable output and
watcher logs remain uncommitted. Standard persistent guards/probe are inherited
from accepted native-control machinery; independent review remains required.
