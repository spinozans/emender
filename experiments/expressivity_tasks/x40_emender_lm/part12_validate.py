"""Audit raw receipts without converting an open performance gate into success."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receipts/part12'


def load(name):
    return json.loads((OUT / name).read_text())


def main():
    support = load('support_gate.json')
    assert support['status'] == 'passed' and len(support['rows']) == 80
    for row in support['rows']:
        assert row['never_touched_nonzero'] == 0 and row['rollback_payload_strength_zero_bits']
        assert all(not field['unjournaled_nonzero_elements'] for field in row['coverage'].values())
        if row['fixture'] == 'correction':
            assert row['corrections'] > 0
    assert load('source_audit.json')['status'] == 'passed'
    sha = hashlib.sha256((ROOT / 'backward.py').read_bytes()).hexdigest()
    for fixture in ('', '_empty', '_popped', '_decay', '_correction'):
        parity = load(f'parity{fixture}.json')
        assert parity['status'] == 'passed' and len(parity['rows']) == 8
        assert parity['sources']['backward.py'] == sha
        assert all(row['status'] == 'passed' and not row['failures'] for row in parity['rows'])
    split_gate = load('parity_split_random.json')
    fusion_gate = load('parity_fused_forward_random.json')
    for proof in (split_gate, fusion_gate):
        assert proof['status'] == 'passed' and len(proof['rows']) == 8
        assert proof['sources']['backward.py'] == sha
    for name in ('part12_decompose.py', 'part12_split_generated.py'):
        assert split_gate['sources'][name] == hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for attempt in range(3):
        assert load(f'parity_split_random_attempt{attempt}.json')['status'] == 'failed'
    established = []
    missing = []
    for context in (512, 2048):
        paths = [OUT/f'curve_{arm}_T2048.json' for arm in ('base', 'candidate')]
        if not all(path.exists() for path in paths):
            missing.append(f'paired curve T{context}')
            continue
        base, candidate = [json.loads(path.read_text()) for path in paths]
        if base['status'] != 'passed' or candidate['status'] != 'passed':
            missing.append(f'failed/incomplete paired curve T{context}')
            continue
        assert min(base['tokens'], candidate['tokens']) >= 2_000_000
        assert base['rows'] == candidate['rows']
        assert base['initial_parameters_sha256'] == candidate['initial_parameters_sha256']
        assert base['final_parameters_sha256'] == candidate['final_parameters_sha256']
        assert base['data_sha256'] == candidate['data_sha256']
        assert base['sources']['backward.py'] == candidate['sources']['backward.py'] == sha
        if context == 2048:
            established.append('1.3B paired 2M-token curve T2048, all update gradient hashes and final parameters equal')
        region_path = OUT/f'split/regions_T{context}.json'
        if region_path.exists():
            region = json.loads(region_path.read_text())
            assert region['reverse_calls'] == 192 and region['split_launches'] == 192*context*4
            assert len(region['raw_reverse_call_seconds']) == 192
            for i, total in enumerate(region['seconds']):
                assert abs(total-sum(row[i] for row in region['raw_reverse_call_seconds'])) < 1e-8
            assert region['reverse_wrapper_overhead_seconds'] >= 0
            assert abs(sum(region['exclusive_update_seconds'].values()) + region['unattributed_seconds'] - region['wall_seconds']) < 1e-7
            established.append(f'additive synchronized split wall T{context}; separate implementation, not original fused timing')
        else:
            missing.append(f'split wall T{context}')
        for arm in ('pure', 'transformer'):
            path = OUT/f'profile/profile_{arm}_T{context}.json'
            if path.exists():
                profile = json.loads(path.read_text())
                assert profile['status'] == 'passed'
                assert abs(sum(profile['exclusive_seconds'].values()) + profile['unattributed_seconds'] - profile['wall_seconds']) < 1e-7
                assert all(value >= 0 for value in profile['exclusive_seconds'].values())
                if arm == 'pure':
                    assert profile['calls']['sequential_forward_core'] == 384
                    assert profile['calls']['forward_read_elementwise_tiles'] == 384
                    assert profile['calls']['reverse_fused_read_math_and_adjoints'] == 192
                established.append(f'{arm} synchronized diagnostic T{context}')
            else:
                missing.append(f'{arm} diagnostic T{context}')
            path = OUT/f'sustained/attempts/1.3B_{arm}_B64_T{context}/00.json'
            if path.exists():
                sustained = json.loads(path.read_text())
                assert sustained['status'] == 'passed'
                assert sustained['warmups'] == 3 and len(sustained['seconds']) == 20
                expected = 64*context*20/sum(sustained['seconds'])
                assert abs(expected/sustained['tokens_per_second'] - 1) < 1e-12
                established.append(f'{arm} 3-warmup/20-update sustained T{context}')
            else:
                missing.append(f'{arm} sustained T{context}')
    archive = OUT/'measured_sources'
    inherited = {OUT/'pilot'/name for name in ('production_T512.json', 'production_T512_expandable.json', 'fused_pool_parity.json')}
    for receipt in inherited:
        assert receipt.read_bytes() == (ROOT/'receipts/part8'/receipt.name).read_bytes()
    inherited_curve = load('pilot/inherited_curve.json')
    original_curve = ROOT/inherited_curve['path']
    assert hashlib.sha256(original_curve.read_bytes()).hexdigest() == inherited_curve['sha256']
    assert (OUT/'pilot/curve/summary.json').read_bytes() == original_curve.read_bytes()
    for receipt in OUT.rglob('*.json'):
        if receipt in inherited:
            continue
        payload = json.loads(receipt.read_text())
        if not isinstance(payload, dict) or not isinstance(payload.get('sources', {}), dict):
            continue
        for name, digest in payload.get('sources', {}).items():
            archived = archive/f'{name}.{digest}.gz'
            assert archived.exists(), (receipt.name, name, digest)
            assert hashlib.sha256(gzip.decompress(archived.read_bytes())).hexdigest() == digest
    for context in (512, 2048):
        normal = load(f'profile/profile_pure_T{context}.json')
        fused = load(f'fused-forward/profile_pure_T{context}.json')
        assert fused['status'] == 'passed' and fused['loss'] == normal['loss']
        assert fused['calls'].get('forward_read_elementwise_tiles', 0) == 0
        assert fused['calls']['sequential_forward_core'] == 384
        assert fused['calls']['reverse_fused_read_math_and_adjoints'] == 192
        for mode in ('profile', 'sustained'):
            paths = [f'profile/profile_{arm}_T{context}.json' if mode == 'profile' else f'sustained/attempts/1.3B_{arm}_B64_T{context}/00.json' for arm in ('pure', 'transformer')]
            pair = [load(path) for path in paths]
            assert pair[0]['physical_gpu'] == pair[1]['physical_gpu']
    pilot = load('pilot/fresh_gate.json')
    assert pilot['status'] == 'passed' and pilot['warmups'] == 3 and len(pilot['seconds']) == 20
    expected = 128*512*20/sum(pilot['seconds'])
    assert abs(expected/pilot['tokens_per_second']-1) < 1e-12
    pilot_gate = load('pilot_gate.json')
    assert pilot_gate['status'] == ('passed' if expected >= 100000 else 'failed')
    assert not missing, missing
    assert '6 passed' in (OUT/'cpu_tests.log').read_text()
    assert not list(OUT.rglob('*.pt'))
    result = dict(status='passed', meaning='Receipt consistency only; not task/performance acceptance',
                  initialization_fixtures=80, native_byte_cases=40, split_byte_cases=8, combined_core_read_byte_cases=8, rejected_split_attempts=3, established=established,
                  missing_or_failed=missing, pilot_tokens_per_second=expected, pilot_100k_gate=pilot_gate['status'],
                  source_archive_scope='All new measured sources; inherited pilot qualifiers checked bytewise against identified part8 provenance',
                  cpu_tests='6 passed', no_pt_files=True)
    (OUT/'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
