"""Freeze X9 prediction and regime identities before any training launch."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket

root = Path('experiments/expressivity_tasks/results_invertible_stage9_anneal_8m')
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

old = json.loads(Path('experiments/expressivity_tasks/results_invertible_stage8_paydepth_8m/registration.json').read_text())
sources = [p for p in old['sources'] if not any(s in p for s in ('stage8_paydepth', 'verify_launch.py'))]
sources += ['experiments/expressivity_tasks/run_invertible_stage9_anneal_8m.py',
            'experiments/expressivity_tasks/launch_invertible_stage9_anneal_8m.sh',
            'scripts/watch_invertible_stage9_anneal.py',
            'tests/test_invertible_stage9_anneal_8m.py',
            str(root / 'receipts/verify_launch.py'), str(root / 'receipts/register.py')]
existing = []
for stage in ('stage6_truecur', 'stage7_deepfirst', 'stage8_paydepth'):
    for p in sorted(Path(f'experiments/expressivity_tasks/results_invertible_{stage}_8m/receipts').glob('*/trainer.json')):
        identity = json.loads(p.read_text())
        proc = Path(f"/proc/{identity['pid']}/stat")
        if not proc.exists():
            continue
        stat = proc.read_text().rsplit(') ', 1)[1].split()
        if stat[0] not in ('Z', 'X') and stat[19] == identity['starttime']:
            existing.append({'identity_path': str(p), 'identity': identity})
(root / 'receipts/existing_runs_before.json').write_text(json.dumps(existing, indent=2) + '\n')
reg = {
    'task': 'X9', 'registered_utc': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
    'approved_scope_amendment': 'Supervisor ruling: REGIME SWAP, not schedule bolted onto ScheduleFree. Ordinary AdamW, standard betas, matching nominal weight decay, warmup/cosine. Everything else X7 bf16 unchanged.',
    'control': 'X7 e97_bf16_seed0: AdamWScheduleFree constant1e-3, betas(.9,.95), decay.01. X6 best-vs-final degradation already occurred UNDER ScheduleFree, weakening a plain oscillation reading.',
    'predictions': [
        '(a) Optimizer-regime structural reading: ordinary AdamW plus anneal permits consolidation once a partial-carry circuit is found. Any driver position >=2 reaching >=50/512 counts as material against X7 flat512,2,0,...; report exact P1-P16 columns.',
        '(b) If still flat, the tested optimizer-regime bundle fails to unlock basin entry. Fixed-lr/ScheduleFree is weakened as the explanation, exhausting the planned 8M probes in this series; next interrogations are scale (1.3B deep-first and ladder-then-deep) and supervision-schedule redesign (proven unlock, current wean fails depth4+). This single seed does not prove all optimizers fail.',
        'Honest asymmetry: anneal plausibly helps CONSOLIDATION, not basin ENTRY. Partial-P2-only/small-P2 evidence is consolidation-only, not proof of a deep entry unlock.',
        'Attribution between anneal, ScheduleFree removal, and standard-beta change remains unresolved: a registered follow-up decomposition is required; no schedule-only causality claim.'
    ],
    'runs': ['e97_bf16_seed0'], 'gpu_binding': {'e97_bf16_seed0': 2}, 'seed': 0, 'depths': [16],
    'objective': 'uniform mean cross entropy over all driver targets; unweighted validation',
    'intervention': 'one optimizer REGIME bundle vs X7 SF-constant control: class, betas and explicit schedule',
    'optimizer': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01,
    'weight_decay_comparison': 'Closest practical nominal matching .01, not exact equivalence: SF adds decay*y after variance normalization before averaged y/z updates; AdamW applies decoupled lr*.01 parameter shrink. Integrated shrink differs under anneal.',
    'lr': .001, 'min_lr': .00001, 'warmup': 5000,
    'schedule': 'update1:2e-7; linear step/5000 through5000:1e-3; cosine (step-5000)/495000, terminal500000:1e-5; actual LR logged every update',
    'clip': 1., 'batch': 32, 'safety_ceiling': 500000, 'eval_every': 500, 'per_position_examples': 512,
    'ladder': False, 'carry_supervision': False, 'stagnation_kill': False, 'threshold_stop': False,
    'initialization': 'scratch, identical X7 bf16 seed0 weights, empty ordinary AdamW state, no checkpoint loaded',
    'validation': 'same frozen X3/X6/X7 seed0 T16 pool; archive/array hashes; unweighted P1-P16 joint/x/y/loss each500 updates',
    'retention': 'X6D immutable entry + best + latest/exit; below400GiB prune other checkpoints with hash manifests; no historical deletion',
    'eval_backend': 'X7 E97 fused_inference=True; legacy recurrent precision; bf16 autocast and fp32 parameters; ordinary unaveraged weights',
    'authority': 'Local single-card expressivity only; R16 evidence discipline, no resilient DiLoCo/Slurm/scale acceptance claimed.',
    'sources': {p: digest(p) for p in sources},
    'preflight_receipts': {p: digest(root / p) for p in ('receipts/tests_prelaunch.txt', 'receipts/tests_exit_code.txt', 'receipts/tests_cpu.txt', 'receipts/existing_runs_before.json')},
    'watcher': 'Parent-owned native supervisor arms scripts/watch_invertible_stage9_anneal.py --interval300 after verified launch; growth report on wake.'
}
(root / 'registration.json').write_text(json.dumps(reg, indent=2) + '\n')
print(f'registered {len(sources)} sources; {len(existing)} live X6/X7/X8 trainers observed, untouched')
