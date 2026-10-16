"""Attended post-completion prune under the queued operator capacity ruling.

Does not modify any X2 evidence bytes; its new manifest is written in X2b.
Fails closed if either root has an active training/evaluation process.
"""
import datetime
import json
from pathlib import Path
import shutil
import subprocess

from experiments.expressivity_tasks.run_invertible_stage2 import MATCHED_STEPS, digest, selected, write_json

base = Path('experiments/expressivity_tasks')
x2, x2b = base / 'results_invertible_stage2', base / 'results_invertible_stage2b'
processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
active = [line for line in processes.splitlines()
          if (' -m experiments.expressivity_tasks.run_invertible_stage2 ' in line
              or ' -m experiments.expressivity_tasks.eval_invertible_matched ' in line)]
if active:
    raise RuntimeError(f'Active training/evaluation processes; prune refused: {active}')

def evidence_hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix != '.pt'}

before_evidence = evidence_hashes(x2)
initial_free = shutil.disk_usage(x2b).free
registered_stops = {json.loads(p.read_text())['stop_step'] for p in x2b.glob('*/selection_frozen.json')}
manifest = {'registered_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'authority': 'Queued parent/operator ruling: immediate capacity prune of both roots; X2 retains only best-val and stop weights, evidence bytes unchanged.',
            'initial_free_bytes': initial_free, 'first_observed_free_bytes': 310896623616,
            'interrupted_attempt': 'Initial full retained-weight rehash timed out after600s; per-run X2b prune receipts were persisted. Resumed using committed pre-prune histories for exact total inventory and existing checkpoint SHA ledgers; selected/stop files freshly rehashed.',
            'active_training_eval_processes': active,
            'broadcast_x2b_stop_steps': sorted(registered_stops), 'runs': {}, 'pruned_bytes': 0}

for root in (x2b, x2):
    for result_path in sorted(root.glob('*/result.json')):
        run = result_path.parent
        result = json.loads(result_path.read_text())
        if result['status'] != 'complete':
            raise RuntimeError(f'Non-complete run: {run}')
        stop = result['stop_step']
        if root == x2b:
            history = json.loads((run / 'validation.json').read_text())
            best, chain = None, set()
            for row in history:
                if selected(row, best):
                    best = row
                    chain.add(row['step'])
            required = chain | MATCHED_STEPS | registered_stops | {stop}
            rule = 'best-val chain +10K +legacy fixed +broadcast stops +latest/own stop'
        else:
            required = {result['step'], stop}
            rule = 'only selected best-validation and own stop'
        ledger = json.loads((run / 'checkpoints.json').read_text())
        # Reconstruct the starting inventory from the pre-prune commit, so
        # deletions from the interrupted first attempt remain fully receipted.
        historically_pruned = set()
        if root == x2b:
            previous = subprocess.run(['git', 'show', f'104ad7ba:{run}/pruning.jsonl'],
                                      text=True, capture_output=True)
            if previous.returncode == 0:
                for line in previous.stdout.splitlines():
                    historically_pruned.update(json.loads(line)['pruned_steps'])
        initial_steps = {int(key) for key in ledger} - historically_pruned
        pruned, retained = [], {}
        for step in sorted(initial_steps):
            path = run / f'step_{step:06d}.pt'
            cp = ledger[str(step)]
            if step in required:
                if not path.exists():
                    # Previously pruned unknown future stop cells stay absent.
                    raise RuntimeError(f'Required starting checkpoint missing: {path}')
                if step in (result['step'], stop) and digest(path) != cp['sha256']:
                    raise RuntimeError(f'Checkpoint identity mismatch: {path}')
                retained[str(step)] = {'sha256': cp['sha256'], 'bytes': path.stat().st_size,
                                       'path': str(path),
                                       'hash_source': 'fresh rehash' if step in (result['step'], stop) else 'original checkpoint SHA256 ledger'}
            else:
                if path.exists():
                    path.unlink()
                pruned.append({'step': step, 'bytes': cp['bytes'], 'sha256': cp['sha256']})
                manifest['pruned_bytes'] += cp['bytes']
        if str(result['step']) not in retained or str(stop) not in retained:
            raise RuntimeError(f'Lost selected or stop checkpoint: {run}')
        event = {'timestamp': manifest['registered_at'], 'at_step': stop,
                 'operator_immediate_capacity_check': True,
                 'pruned_steps': [p['step'] for p in pruned],
                 'retained_hashes': {s: c['sha256'] for s, c in retained.items()},
                 'known_stop_steps': sorted(registered_stops),
                 'inventory_baseline_commit': '104ad7ba',
                 'includes_deletions_from_interrupted_attempt': True}
        if root == x2b:
            with (run / 'pruning.jsonl').open('a') as handle:
                handle.write(json.dumps(event) + '\n')
        manifest['runs'][str(run)] = {'rule': rule, 'stop_step': stop,
                                     'required_steps': sorted(s for s in required if s <= stop),
                                     'retained': retained, 'pruned': pruned,
                                     'pruned_bytes': sum(p['bytes'] for p in pruned)}
        write_json(x2b / 'receipts/postcompletion_capacity_prune.json', manifest)

# Failed/aborted X2 alignment attempts have no final selection/stop contract.
# Their four existing weights are left intact, and explicitly inventoried.
manifest['x2_archived_alignment_weights_unchanged'] = {
    str(p): {'sha256': digest(p), 'bytes': p.stat().st_size}
    for p in sorted((x2 / 'attempt1_alignment').rglob('*.pt'))}
assert evidence_hashes(x2) == before_evidence
manifest['x2_non_checkpoint_evidence_byte_identical'] = True
manifest['x2_evidence_sha256'] = before_evidence
manifest['final_free_bytes'] = shutil.disk_usage(x2b).free
manifest['observed_free_increase_bytes'] = manifest['final_free_bytes'] - manifest['first_observed_free_bytes']
write_json(x2b / 'receipts/postcompletion_capacity_prune.json', manifest)
print(json.dumps({k: manifest[k] for k in ('pruned_bytes', 'initial_free_bytes', 'final_free_bytes',
                                           'observed_free_increase_bytes', 'x2_non_checkpoint_evidence_byte_identical')}, indent=2))
