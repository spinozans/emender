"""Register the approved X9b schedule-under-SF cell before launch."""
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import schedulefree
import socket

from experiments.expressivity_tasks.run_invertible_stage9b_sf_anneal_8m import digest, write_json, ROOT
from scripts.watch_invertible_stage9b_sf_anneal import process_alive

old_root = ROOT.parent / 'results_invertible_stage9_anneal_8m'
old = json.loads((old_root / 'registration.json').read_text())
prior = {}
for stage in ('results_invertible_stage6_truecur_8m', 'results_invertible_stage7_deepfirst_8m'):
    for p in sorted((ROOT.parent / stage / 'receipts').glob('*/trainer.json')):
        identity = json.loads(p.read_text())
        prior[str(p)] = {'identity': identity, 'alive': process_alive(identity)}
write_json(ROOT / 'receipts/existing_runs_before.json', prior)
reg = {k: v for k, v in old.items() if k not in ('sources', 'preflight_receipts', 'approved_scope_amendment')}
reg.update({
    'task': 'X9b', 'registered_utc': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
    'control': 'X7 E97 bf16 SF-constant (.9,.95); X9 E97 bf16 ordinary AdamW+anneal (.9,.999)',
    'predictions': [
        '(i) If X9b composes (any position P2+ reaching >=50/512), the SCHEDULE carries the unlock and SF was masking it: external scheduling suffices even while SF averaging is retained.',
        '(ii) If X9b stays flat like X7 (no P2+ reaching >=50/512), the SCHEDULE-FREE CLASS is the binding-structure hypothesis: its averaging/anchor dynamics gate basin entry; the remaining AdamW+constant cell, queued for the next free card, is the confirming follow-up, not launched here.',
    ],
    'interpretation_limits': 'Predictions are registered mechanistic hypotheses, not proof of a unique cause. One seed, one depth, one regime. X9b vs X7 isolates external scheduling within SF, keeping X7 betas(.9,.95). X9 vs X9b still changes beta2 .999/.95 as well as class/decay semantics, so a negative X9b does not alone isolate averaging or rule out beta effects. The remaining AdamW+constant cell is queued only; its launch is not authorized by this script.',
    'intervention': 'schedule-under-SF cell; X7 class, betas, averaging, eps and decay retained; only explicit warmup/cosine added',
    'optimizer': 'AdamWScheduleFree', 'betas': [.9, .95], 'internal_warmup': 0,
    'weight_decay_comparison': 'X7 SF decay-at-y .01 retained; integrated shrink changes with schedule. Not equivalent to ordinary AdamW decay.',
    'initialization': 'scratch, exact X7 bf16 seed0 model, empty ScheduleFree state; no checkpoint loaded',
    'averaging': 'retained SF train y/eval x; r=0, weight_lr_power=2, lr_max weighting unchanged',
    'eval_backend': 'X7 sequential Triton E97 fused_inference=True, legacy recurrent precision; SF eval x averaged weights',
    'actual_lr_logging': 'requested external LR AND optimizer scheduled_lr logged each update; internal warmup_steps=0 prevents double warmup',
    'watcher': 'Supervisor must arm scripts/watch_invertible_stage9b_sf_anneal.py --interval 300 after verified launch. Watcher is not armed by launcher.',
    'installed_sources': {inspect.getfile(schedulefree.AdamWScheduleFree): digest(inspect.getfile(schedulefree.AdamWScheduleFree))},
})
sources = [p for p in old['sources'] if 'stage9_anneal' not in p and '/receipts/' not in p]
sources += ['experiments/expressivity_tasks/run_invertible_stage9b_sf_anneal_8m.py',
    'experiments/expressivity_tasks/launch_invertible_stage9b_sf_anneal_8m.sh',
    'scripts/watch_invertible_stage9b_sf_anneal.py', 'tests/test_invertible_stage9b_sf_anneal_8m.py',
    str(ROOT / 'receipts/register.py'), str(ROOT / 'receipts/verify_launch.py')]
reg['sources'] = {p: digest(p) for p in sources}
reg['preflight_receipts'] = {f'receipts/{p}': digest(ROOT / 'receipts' / p) for p in (
    'tests_prelaunch.txt', 'tests_exit_code.txt', 'tests_cpu.txt', 'existing_runs_before.json')}
write_json(ROOT / 'registration.json', reg)
print(reg['registered_utc'])
