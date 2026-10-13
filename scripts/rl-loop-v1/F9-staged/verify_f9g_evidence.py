"""Read-only F9G exact-byte census and sealed/additive archive qualification.

Run after sourcing the evidence's private source closure activate.sh. No live
admission, deployment, bank access or candidate command execution occurs.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import tarfile

from ndm.e97_first_party_source_archive import _manifest_components_payload, verify_archive_members

REPO = Path('/home/erikg/emender')
S = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = json.loads((S / 'F9-evidence.json').read_bytes())
    root = Path(evidence['source_closure_root'])
    assert Path(os.environ['F9_STAGE_ROOT']).resolve() == root
    groups = [(evidence, 'staged_source_files', REPO),
              (evidence, 'integrated_runtime_files', root),
              (evidence, 'private_artifacts', Path(evidence['private_evidence_root'])),
              (evidence, 'qualification_logs', Path('/'))]
    for repair in ('f9e', 'f9g'):
        for category in ('regression_files', 'qualification_logs', 'reviewer_artifacts'):
            if category in evidence[repair]:
                groups.append((evidence[repair], category, REPO if category == 'regression_files' else Path('/')))
    count = 0
    for section, category, base in groups:
        for name, expected in section[category].items():
            path = base / name
            assert sha(path) == expected, (category, name)
            count += 1
        print(category, len(section[category]), 'pins MATCH')
    print(count, 'file-pin occurrences MATCH (historical live-source hashes are not current-source claims)')
    reviewer_path = next(Path(name) for name in evidence['f9g']['reviewer_artifacts']
                         if name.endswith('/F9R3-grammar-results.json'))
    reviewer_cases = json.loads(reviewer_path.read_bytes())['cases']
    adopted = [{key: case[key] for key in ('category', 'command', 'expected')}
               for case in reviewer_cases if case['category'].startswith('new/')]
    assert adopted == json.loads((S / 'F9R3-grammar-cases.json').read_bytes())['cases']
    print('237 adopted cases exactly MATCH pinned original F9R3 reviewer results')
    generated = runpy.run_path(str(S / 'sync_validator.py'))['validator_source']().encode()
    validator = root / 'scripts/e97_first_party_validator_diversity.py'
    assert generated == validator.read_bytes() == (S / validator.relative_to(root)).read_bytes()
    assert sha(validator) == evidence['sealed_validator_sha256']
    assert runpy.run_path(str(S / 'prepare.py'))['EXPECTED_VALIDATOR_SHA256'] == sha(validator)
    for directory in ('scripts', 'ndm', 'loop', 'configs'):
        for path in (S / directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                assert path.read_bytes() == (root / path.relative_to(S)).read_bytes(), path
    print('In-memory regeneration, prepare pin and staged/runtime bytes MATCH')
    for name in ('rl_correction_service.py', 'rl_correction_pool.py', 'rl_correction_config.py', 'rl_bank.py'):
        assert (root / 'loop' / name).read_bytes() == (REPO / 'scripts/rl-loop-v1' / name).read_bytes()
    old = REPO / 'configs/pi/e97-firstparty-generator-manifest-v1.json'
    manifest = root / 'configs/pi/e97-firstparty-generator-manifest-v1.json'
    archive = root / 'configs/pi/e97-firstparty-source-v1.tar'
    _manifest_components_payload(old.read_bytes())
    _manifest_components_payload(manifest.read_bytes())
    assert len(json.loads(old.read_bytes())['components']) == 17
    additive = json.loads(manifest.read_bytes())
    assert len(additive['components']) == 18
    additive['components'].append({'path': 'scripts/unauthorized.py', 'sha256': 'a' * 64})
    try:
        _manifest_components_payload(json.dumps(additive, sort_keys=True, separators=(',', ':')).encode())
    except ValueError:
        pass
    else:
        raise AssertionError('extra closure component admitted')
    assert verify_archive_members(manifest, archive) == evidence['source_archive_sha256'] == sha(archive)
    registry = json.loads((root / 'configs/pi/e97-onpolicy-source-registry-v1.json').read_bytes())
    pins = [source['receipts']['source_archive_sha256'] for source in registry['sources']
            if source['kind'] == 'first-party' and source['status'] == 'admitted']
    assert pins == [sha(archive)] * 2
    with tarfile.open(archive) as tar:
        for component in json.loads(manifest.read_bytes())['components']:
            name = component['path']
            assert tar.extractfile(name).read() == (root / name).read_bytes()
        for name in ('e97_first_party_validator.py', 'e97_first_party_validator_first_action.py',
                     'e97_first_party_validator_protocol_breadth.py'):
            assert tar.extractfile('scripts/' + name).read() == (REPO / 'scripts' / name).read_bytes()
    print('Current helpers, old17/additive18/extra rejection, both registry pins and all18 archive members PASS')
    print('validator', sha(validator))
    print('archive', sha(archive))


if __name__ == '__main__':
    main()
