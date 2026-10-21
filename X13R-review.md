## Review

No issues found.

- **Correct — split outcome:** terminal/DONE receipts agree; result files confirm both 1M completions and seed2’s final **512/512, 512/512, 309/512**. Native `GROWTH_REPORT.md:7–16` corroborates seed1’s global maximum **4**, seed2’s P3 maximum **321**, and crossings **523K → 661.5K → 840.5K**.
- **Correct — interpretation:** `X13-growth-report.md:3–5,137–151` distinguishes demonstrated can-learn from unreliable finite-recipe ignition, treats **1/3 as descriptive**, and rejects architectural impossibility and deterministic probability claims.
- **Correct — replication distinction:** `X13-growth-report.md:149` explicitly records concession #8. X9’s `e97_bf16_seed0/config.json:4` and X10’s `adamw_seed0/config.json:3` independently confirm **both use seed0**.
- **Correct — censoring and fixes:** `X13-growth-report.md:36–38,153–167` separates first crossing from permanent retention, absent events from impossibility, inherited curriculum retention from new acquisition, and transient carry-wean success from final survival. Descriptions match X16/X17 registrations.
- **Correct — investigation continuity:** `X13-report.md:5,168–172` clearly supersedes historical pending language without rewriting launch evidence.

**Verdict: ACCEPT-WITH-CONDITIONS.**  
**Exact condition:** “Before closing acceptance, the parent must successfully run and retain output from the read-only CPU audits below, confirming the detailed history-derived numbers and reported artifact hashes. Any discrepancy must be reconciled before acceptance. No new training or GPU evaluation is required.”

**Merge verdict: OK with notes**, subject to that validation condition.

### Requested CPU audits

Run from `/home/erikg/emender` on lambda01. On Frontier, first activate the canonical environment and substitute `"$EMENDER_PYTHON"`.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/audit_x13_growth.py
```

I inspected that existing script: it checks terminal binding, complete validation grids, aggregate/best/prefix identities, native table regeneration, and both complete finite training/LR streams. Compare its six printed hashes with `X13-growth-report.md:179–181`.

Then independently verify the report-specific timing and table claims:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import json, math
from pathlib import Path
from scripts.watch_invertible_stage13_zerocheck import DEFAULT_ROOT as r, NAMES

a, b = [json.loads((r/n/'validation.json').read_text()) for n in NAMES]
assert max(p['correct'] for x in a for p in x['positions']) == 4
assert max(x['positions'][0]['correct'] for x in a) == 3
assert [p['correct'] for p in b[-1]['positions'][:3]] == [512,512,309]
assert max(x['positions'][2]['correct'] for x in b) == 321
assert next(x['step'] for x in b if x['positions'][2]['correct']==321) == 996500
assert max(p['correct'] for x in b for p in x['positions'][3:]) == 4

for j, first50, first512, dips, lastdip in [
    (0,523000,540000,110,733000),
    (1,661500,726500,22,783500),
    (2,840500,None,None,None),
]:
    for threshold, expected in [(50,first50),(512,first512)]:
        assert next((x['step'] for x in b
                     if x['positions'][j]['correct']>=threshold),None)==expected
    if dips is not None:
        later=[x for x in b if x['step']>first512
               and x['positions'][j]['correct']<512]
        assert (len(later),later[-1]['step'])==(dips,lastdip)
assert next(x['step'] for x in b if x['perfect_prefix_depth']>=2)==726500

losses=[x['aggregate']['loss'] for x in a if x['step']>=50000]
assert abs(min(losses)-8.221666038)<1e-9
assert abs(max(losses)-8.222116947)<1e-9
assert abs(a[-1]['aggregate']['loss']-math.log(3721))<4e-7

def positional_lines(path):
    return [s for s in path.read_text().splitlines()
            if s.startswith('| P') and s.split('|')[1].strip()[1:].isdigit()]
assert positional_lines(Path('X13-growth-report.md')) == \
       positional_lines(r/'GROWTH_REPORT.md')
print('PASS: lottery counts, escape timing, retention dips, marginal CE, 64 printed rows')
PY
```

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m pytest -q tests/test_invertible_stage13_zerocheck.py -k 'not gpu'
```

**Review limitations:** the requested documents actually reside at repository root, not under the results directory. Neither has a working-tree delta against reviewer-launch HEAD; this reviews their current contents, not a committed range. Raw streams remain local/untracked. No write-capable tool was available; this final response is the complete review artifact for persistence.
---

## Parent attestation (2026-10-21)

X13R condition satisfied. Audits run by the parent on lambda01:

1. `PYTHONPATH=. CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/audit_x13_growth.py` - PASS both seeds
   (2001 validations each; 1M finite consecutive training/LR/gradient rows; result/best/terminal
   identities; native terminal status + full-position growth table regenerated). All six printed
   artifact hashes match X13-growth-report.md:179-181 exactly.
2. Inline history-derivation audit (reviewer text verbatim) - PASS: lottery counts, escape timing
   (first-50: 523000/661500/840500; first-512: 540000/726500; dips 110/22 with last 733000/783500;
   perfect-prefix-depth>=2 at 726500), marginal CE bounds [8.221666038, 8.222116947] and final
   loss = ln(3721), P3 max 321 at step 996500, and the 64 positional table rows identical to the
   generated GROWTH_REPORT.md.
3. `pytest -q tests/test_invertible_stage13_zerocheck.py -k 'not gpu'` - 11 passed.

Verdict: X13R ACCEPT-WITH-CONDITIONS -> ACCEPTED. X13 closed.
