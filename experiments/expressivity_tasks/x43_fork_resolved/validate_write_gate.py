"""CPU receipt replay and paired comparisons for the preregistered gate probe."""
import json
import numpy as np
from .write_gate import ROOT, OUT, ARMS, sha, dump, metrics


def main():
    reg = json.loads((OUT/'registration.json').read_text())
    for name, digest in reg['artifacts'].items():
        assert sha(ROOT/name) == digest, name
    results = json.loads((OUT/'results.json').read_text())
    assert results['registration_sha256'] == sha(OUT/'registration.json')
    assert [r['arm'] for r in results['models']] == list(ARMS)
    assert json.loads((OUT/'terminal.json').read_text())['exit_code'] == 0
    assert json.loads((OUT/'tests.json').read_text())['status'] == 'passed'
    fork = np.load(ROOT/'receipts/part1c/labels.npz')['fork']
    split = int(.9*len(fork)); pairs = []
    for row in results['models']:
        assert row['trainable_parameters'] == 2000324
        assert row['train_seconds'] >= 180 and row['steps'] >= 1000
        for suffix, key in (('', 'evaluation'), ('_step1000','matched_step1000')):
            name = row['arm']+suffix
            ev = np.load(OUT/f'{name}_per_position.npz')
            assert np.array_equal(ev['position'], np.arange(split+1,len(fork)))
            recomputed = metrics(ev['nll'],ev['correct'],ev['position'],fork)
            assert recomputed == row[key]['full']
            assert row[key]['forbidden_writes'] == 0
            for layer in row[key]['telemetry']:
                for head in layer:
                    assert head['event_density'] <= row[key]['eligibility_density']+1e-7
            base = np.load(OUT/f'baseline{suffix}_per_position.npz')
            delta = ev['nll'].astype(np.float64)-base['nll']
            mask = fork[ev['position']]
            pairs.append(dict(arm=name, overall_delta=float(delta.mean()),
                              fork_delta=float(delta[mask].mean()),
                              forced_delta=float(delta[~mask].mean()),
                              joint_prediction_passed=bool(delta[mask].mean()<0 and delta[~mask].mean()<=0)))
    dump(OUT/'validation.json', dict(status='passed', checks=[
        'preregistration hashes unchanged', 'four original 180s protocol arms completed',
        'full held-out contiguous target coverage', 'NLL decomposition replay exact',
        'zero forbidden write events in every evaluation scan',
        'event densities bounded by eligibility', 'matched step1000 comparisons complete'],
        comparisons=pairs))
    manifest = {str(p.relative_to(OUT)):dict(sha256=sha(p), bytes=p.stat().st_size)
                for p in sorted(OUT.iterdir()) if p.is_file() and p.suffix != '.pt' and p.name != 'manifest.json'}
    dump(OUT/'manifest.json', manifest)
    print(json.dumps(pairs, indent=2))


if __name__ == '__main__':
    main()
