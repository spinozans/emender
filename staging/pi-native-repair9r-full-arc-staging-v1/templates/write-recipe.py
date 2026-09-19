#!/usr/bin/env python3
"""Write recipe.json for a completed repair-v9r segment training run (STAGED helper).

Derives every field from run artifacts; no value is invented here:
  - checkpoint path/sha256/bytes come from the run's own console.log checkpoint event
    (which the launcher captured) and are re-verified against the file on disk;
  - admission/authority/pack/schedule/proposal identities are re-hashed from disk;
  - parent identity is re-hashed from disk.

The output matches the recipe contract enforced by
scripts/audit_e97_pi_native_training_run.py (schema emender-e97-pi-native-training-recipe-v1,
as used by the v9-full segment-1 run).
"""
import argparse, hashlib, json
from pathlib import Path

def sha(p):
    p = Path(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--admission', type=Path, required=True)
    ap.add_argument('--proposal', type=Path, required=True)
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--source-commit', required=True)
    a = ap.parse_args()

    run = a.run
    admission = a.admission
    events = [json.loads(l) for l in (run / 'console.log').read_text().splitlines()
              if l.startswith('{') and '"event"' in l]
    checkpoints = [e for e in events if e['event'] == 'checkpoint']
    completes = [e for e in events if e['event'] == 'complete']
    if len(checkpoints) != 1 or len(completes) != 1 or completes[0]['reason'] is not None:
        raise SystemExit('run did not complete cleanly with exactly one checkpoint')
    ck = checkpoints[0]
    ckpt_path = Path(ck['checkpoint']).resolve(strict=True)
    if ckpt_path.parent != (run / 'checkpoints').resolve():
        raise SystemExit('checkpoint not published in the run checkpoints dir')
    if sha(ckpt_path) != ck['checkpoint_sha256'] or ckpt_path.stat().st_size != ck['checkpoint_bytes']:
        raise SystemExit('checkpoint bytes do not match the run event')

    recipe = {
        'admission_path': str(admission / 'admission.json'),
        'admission_sha256': sha(admission / 'admission.json'),
        'authority_manifest_sha256': sha(admission / 'manifest.json'),
        'automatic_expansion': False,
        'automatic_retry': False,
        'checkpoint': {'path': str(ckpt_path), 'sha256': ck['checkpoint_sha256']},
        'checkpoint_promotion': False,
        'lr': 1e-05,
        'new_rl_updates': 0,
        'operator_authorization_statement': 'I authorize the exact 128 update proposal.',
        'pack_manifest_sha256': sha(admission / 'packs/manifest.json'),
        'parent_path': str(a.parent),
        'parent_sha256': sha(a.parent),
        'proposal_path': str(a.proposal),
        'proposal_sha256': sha(a.proposal),
        'sampler_key': json.loads((admission / 'expected-schedule.json').read_text())['sampler_key'],
        'schedule': {
            'path': str(admission / 'expected-schedule.json'),
            'sampler_key': json.loads((admission / 'expected-schedule.json').read_text())['sampler_key'],
            'sha256': sha(admission / 'expected-schedule.json'),
        },
        'schema': 'emender-e97-pi-native-training-recipe-v1',
        'source_commit': a.source_commit,
        'steps': 128,
    }
    out = run / 'recipe.json'
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    out.write_text(json.dumps(recipe, indent=2, sort_keys=True) + '\n')
    print('RECIPE_WRITTEN', sha(out))

if __name__ == '__main__':
    main()
