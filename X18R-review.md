## Review

Let `R = experiments/expressivity_tasks/results_invertible_stage18_freshdata_8m`.

- **Correct — selections:** `R/registration.json:9–16,736–743,1461–1468,2667–2674` freezes the requested four updates and hashes. Historical `result.json` receipts agree: X10/X11 use aggregate-best checkpoints; X12/X13-seed2 use final-1M, not their independently best checkpoints.
- **Correct — ordering evidence:** registration records **20:11:03**, generation begins **20:11:19** (`R/generation.json:2–4`). `R/receipts/manual_verification.json:2–5` attests commit `9445d165` at **20:11:13**. Independent verification of that committed artifact remains a condition below; available tools cannot inspect committed history.
- **Correct — novelty:** generation binds new seeds, archives, arrays and RNG states. All four novelty receipts record zero full-sequence matches across **94,416,000 replayed training sequences**, with checkpoint RNG agreement. The implementation uses full-byte membership after a prefix prefilter (`eval_invertible_stage18_freshdata.py:181–204`), not a prefix-only novelty claim.
- **Correct — evaluation controls:** GPU3 co-residency and no exclusive lease are disclosed (`R/runtime.json:5–14`). Evaluation strictly loads the registered copied checkpoint, checks frozen controls before fresh evaluation, and verifies unchanged tensors/payloads (`eval_invertible_stage18_freshdata.py:242–292`).
- **Correct — outcome:** the decisive counts agree with the evaluation receipts and `R/audit.json:5–46`: X10 **512×4**, X11 **512×4**, X12 **512,512,512,512,510**, X13-seed2 **512×2**. All **15 previously-perfect positions** retain at least **510/512**, exceeding the preregistered **480/512** criterion. Registered **(a), EXECUTING** follows; X12 P5 is correctly described as imperfect.
- **Correct — scope:** `X18-report.md:5,92` explicitly limits this to same-distribution, native-length generalization. It does not claim OOD extrapolation, unlimited composition, circuit identification, or exclusion of finite-prefix lookup mechanisms.

**No issues found.**

**Merge verdict: OK with notes.**  
**Acceptance verdict: ACCEPT-WITH-CONDITIONS.**

**Exact condition:** “Accept X18 after the parent successfully runs and retains the outputs of audits A–C below, confirming manifest integrity, the unchanged preregistration artifact committed before fresh generation, deterministic fresh-pool reconstruction, zero training/frozen-pool overlap, and all 160 report rows. Missing artifacts or failed assertions leave acceptance BLOCKED pending reconciliation.”

### Requested parent CPU audits

Run from the repository root on **lambda01**, using its recorded `.venv`. These do not run GPU evaluation or rewrite X18 receipts. Do **not** invoke the evaluator’s `audit` action: it rewrites `audit.json`.

**A — integrity**
```bash
sha256sum -c experiments/expressivity_tasks/results_invertible_stage18_freshdata_8m/SHA256SUMS
```

**B — bounded preregistration, pools, novelty and table audit**
```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import hashlib, json, math, re, subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from experiments.expressivity_tasks import eval_invertible_stage18_freshdata as x

R = x.ROOT
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg, gen = x.registration(), read(R / 'generation.json')
assert gen['registration_sha256'] == sha(R / 'registration.json')

# Inspect only the named preregistration commit and its three task artifacts.
for path in (
    R / 'registration.json',
    Path('experiments/expressivity_tasks/eval_invertible_stage18_freshdata.py'),
    Path('tests/test_eval_invertible_stage18_freshdata.py'),
):
    committed = subprocess.check_output(['git', 'show', f'9445d165:{path}'])
    assert committed == path.read_bytes(), path
commit_time = subprocess.check_output(
    ['git', 'show', '-s', '--format=%cI', '9445d165'], text=True).strip()
assert (datetime.fromisoformat(reg['registered_utc'])
        < datetime.fromisoformat(commit_time)
        < datetime.fromisoformat(gen['started_utc']))

expected = {
    'X10-adamw': 468000, 'X11-GDN2': 482500,
    'X12-GDN2': 1000000, 'X13-seed2': 1000000,
}
pools = {}
for depth, seed in ((16, 202610211816), (64, 202610211864)):
    info = gen['pools'][str(depth)]
    assert reg['fresh_seeds'][str(depth)] == info['seed'] == seed
    x.require_hash(info['path'], info['archive_sha256'])
    pool = x.load_pool(info['path'], depth)
    rng = np.random.default_rng(seed)
    assert rng.bit_generator.state == info['initial_rng']
    rebuilt = x.fresh_training_batch(
        x.x12.x10.InvertibleQuadraticTask(61), 512, depth, rng)
    assert rng.bit_generator.state == info['final_rng']
    assert x.hashes(pool) == x.hashes(rebuilt) == info['array_sha256']
    assert len(x.row_set(pool[0])) == info['unique_sequences'] == 512
    assert not any(info['frozen_overlap'].values())
    x.validate_pool(pool, depth)
    pools[depth] = pool

report = Path('X18-report.md').read_text()
sections = dict((name.strip(), body) for name, body in re.findall(
    r'^### ([^\n/]+) / T\d+\n(.*?)(?=^### |\Z)', report, re.M | re.S))
rows_checked = 0
for run in reg['champions']:
    name, depth, cp = run['name'], run['depth'], run['checkpoint']
    assert cp['step'] == expected[name]
    for path, digest in run['historical_files_sha256'].items():
        x.require_hash(path, digest)
    original = Path(cp['path']).parent
    result = read(original / 'result.json')
    selection = 'best' if name in ('X10-adamw', 'X11-GDN2') else 'final'
    assert result[selection]['checkpoint'] == cp
    row = next(r for r in read(original / 'validation.json')
               if r['step'] == cp['step'])
    assert {k: row[k] for k in run['frozen_metrics']} == run['frozen_metrics']
    x.require_hash(cp['path'], cp['sha256'])
    x.require_hash(run['snapshot_path'], cp['sha256'])
    x.require_hash(run['frozen_pool'], run['frozen_pool_sha256'])
    frozen = x.load_pool(run['frozen_pool'], depth)
    assert x.hashes(frozen) == run['frozen_array_sha256']
    assert not (x.row_set(frozen[0]) & x.row_set(pools[depth][0]))

    saved = torch.load(run['snapshot_path'], map_location='cpu', weights_only=False)
    matches, state = x.training_overlap(
        pools[depth], run['seed'], depth, cp['step'])
    novelty = read(R / 'novelty' / f'{name}.json')
    assert matches == novelty['full_sequence_matches'] == 0
    assert state == saved['numpy_rng'] == novelty['terminal_rng']
    assert novelty['terminal_rng_matches_checkpoint']
    assert novelty['updates_replayed'] == cp['step']
    assert novelty['sequences_checked'] == cp['step'] * 32
    assert novelty['champion_sha256'] == cp['sha256']
    assert novelty['generation_sha256'] == sha(R / 'generation.json')
    del saved

    receipt = read(R / 'evaluations' / f'{name}.json')
    assert receipt['champion_sha256'] == cp['sha256']
    assert receipt['registration_sha256'] == sha(R / 'registration.json')
    assert receipt['generation_sha256'] == sha(R / 'generation.json')
    assert receipt['training_updates'] == 0
    assert receipt['weights_unchanged'] and receipt['frozen_exact_match']
    assert receipt['frozen'] == run['frozen_metrics']
    retained = [b['correct'] for a, b in zip(
        receipt['frozen']['positions'], receipt['fresh']['positions'])
        if a['correct'] == 512]
    assert min(retained) >= 510
    assert read(R / 'audit.json')['champions'][name][
        'fresh_counts_at_frozen_perfect_positions'] == retained

    lines = [line for line in sections[name].splitlines()
             if re.match(r'^\| P\d+ \|', line)]
    assert len(lines) == depth
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split('|')[1:-1]]
        assert cells[0] == f'P{i+1}'
        for column, key in enumerate(('correct', 'x_correct', 'y_correct', 'loss'), 1):
            displayed = cells[column].split('→')
            for side, value in zip(('frozen', 'fresh'), displayed):
                actual = receipt[side]['positions'][i][key]
                assert math.isclose(float(value), actual, rel_tol=1e-8, abs_tol=1e-10)
        rows_checked += 1
    print(name, retained, 'zero overlap; checkpoint RNG matched')
assert rows_checked == 160
assert read(R / 'audit.json')['verdict'] == 'a_executing'
print('PASS: preregistration, deterministic pools, novelty, 160 positional rows')
PY
```

**C — focused CPU regression tests**
```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_eval_invertible_stage18_freshdata.py tests/test_eval_invertible_stage6_positions.py tests/test_invertible_quadratic.py
```

Residual risks remain those disclosed: one pool per depth, selected champions, same finite distribution, and no mechanism/OOD evidence. GPU execution was inspected through receipts, not independently rerun.
---

## Parent attestation (2026-10-21)

X18R condition satisfied - all three audits run by the parent on lambda01, outputs retained:

A. `sha256sum -c .../SHA256SUMS` - all artifacts OK (every line verified).
B. Reviewer's audit-B script run VERBATIM (extracted from this review's code block; one
   parent-side transcription of the truncated preview initially used a wrong key name
   `generated_at` and failed with KeyError - the reviewer's verbatim script correctly uses
   `started_utc`; re-run verbatim): PASS per champion - X10-adamw [512,512,512,512],
   X11-GDN2 [512,512,512,512], X12-GDN2 [512,512,512,512,510], X13-seed2 [512,512] -
   zero overlap, checkpoint RNG matched, preregistration committed before generation
   (registered_utc < commit 9445d165 < started_utc), deterministic pool rebuild,
   160 positional report rows exact, verdict a_executing.
C. `pytest -q tests/test_eval_invertible_stage18_freshdata.py tests/test_eval_invertible_stage6_positions.py tests/test_invertible_quadratic.py` - 66 passed.

Verdict: X18R ACCEPT-WITH-CONDITIONS -> ACCEPTED. X18 executing-not-memorizing closed.
