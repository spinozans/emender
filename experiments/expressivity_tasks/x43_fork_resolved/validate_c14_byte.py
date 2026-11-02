"""CPU-only registration audit; green is NOT training or query-health acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from c14_documents import sha

ROOT = Path(__file__).resolve().parent


def validate(full_source=False):
    r = json.loads((ROOT/'receipts/c14_byte/registration.json').read_text())
    assert r['claim'] == 'C14' and r['status'] == 'registered'
    assert r['vocabulary'] == 256 and r['no_cross_document_context'] is True
    assert r['training_started'] is False and r['launch_ready'] is False
    assert r['launch_blockers']
    for name, digest in r['files'].items():
        assert sha(ROOT/name) == digest, name
    assert sha(Path(r['wiki_manifest_path'])) == r['wiki_manifest_sha256']
    assert sha(Path(r['xsa_path'])) == r['xsa_sha256']
    assert Path(r['index_path']).stat().st_size == r['index_bytes']
    models = json.loads((ROOT/'receipts/c14_byte/model_shapes.json').read_text())
    assert models['device'] == 'cpu' and not models['training_started']
    assert [m['arm'] for m in models['models']] == ['emender', 'transformer']
    for model in models['models']:
        assert model['shapes']['device_input.weight'] == [256, 260]
        assert model['modes'] == ['dense_zero_input', 'wikipedia_dropout']
    for path, digest in models['sources'].items():
        assert sha(ROOT.parents[2]/path) == digest, path
    discovery = r['source_discovery']
    assert sha(ROOT.parents[2]/discovery['documents_index']) == discovery['documents_index_sha256']
    assert discovery['matched_documents'] == 4096 and discovery['cross_document_windows'] == 0
    if full_source:
        assert sha(r['index_path']) == r['index_sha256']
        assert sha(discovery['raw_source']) == discovery['raw_source_sha256']
    if r['coverage_status'] == 'bound':
        assert len(r['coverage_binding']) == 4
        for row in r['coverage_binding'].values():
            raw = subprocess.check_output(['git', 'show', row['commit']+':'+row['path']], cwd=ROOT)
            assert hashlib.sha256(raw).hexdigest() == row['sha256']
            assert sha(ROOT.parents[2]/row['path']) == row['sha256']
    else:
        assert r['coverage_status'] == 'pending-coverage' and not r['coverage_binding']
    return dict(status='green', scope='registration and boundary preflight only',
                training_ready=False, coverage_status=r['coverage_status'],
                full_raw_source_hash_verified=full_source, blockers=r['launch_blockers'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--full-source', action='store_true')
    print(json.dumps(validate(p.parse_args().full_source), indent=2))
