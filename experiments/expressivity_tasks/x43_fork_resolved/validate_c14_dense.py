"""Independent CPU replay of C14 dense source, evaluation, and loss receipts."""
import hashlib
import json
import math
from pathlib import Path

from .c14_documents import evaluation_halves, sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'receipts/c14_dense'


def validate():
    span = json.loads((OUT/'pile_span.json').read_text())
    launch = json.loads((OUT/'launch.json').read_text())
    construction = json.loads((OUT/'eval_construction.json').read_text())
    training = json.loads((OUT/'training.json').read_text())
    assert span['status'] == 'passed' and span['matched_documents'] == 7831213
    assert span['excluded_token_truncated'] == 2 and span['materialized_bytes'] == 0
    assert span['next_missing_receipt'] is None and span['source_unchanged']
    assert launch['source_span_sha256'] == construction['source_span_sha256'] == sha(OUT/'pile_span.json')
    assert launch['features'] == 'all-zero' and launch['physical_gpu'] in ('2','4')
    assert launch['interpreter'] == '/home/erikg/emender/.venv/bin/python'
    assert (launch['seed'], launch['batch'], launch['context'], launch['budget_seconds_per_head']) == (3401,16,128,2700)
    if training.get('continuation'):
        assert sha(OUT/'training_code_take3.py.txt') == launch['code_sha256']
        continuation=json.loads((OUT/'continuation_launch.json').read_text())
        assert training['original_launch_sha256'] == sha(OUT/'launch.json')
        assert continuation['code_sha256'] == sha(ROOT/'c14_dense_train.py') == training['code_sha256']
        for key in ('source_span_sha256','protocol_sha256','dense_amendment_sha256','model_dependencies','seed','batch','context','budget_seconds_per_head','features'):
            assert continuation[key] == launch[key]
        inventory=json.loads((OUT/'inventory_take4.json').read_text())
        assert training['models'][0]['checkpoint_sha256'] == inventory['arms']['emender']['checkpoint_sha256']
    else:
        assert launch['code_sha256'] == sha(ROOT/'c14_dense_train.py') == training['code_sha256']
    assert launch['protocol_sha256'] == sha(ROOT/'PREREGISTRATION_C14_BYTE.md')
    assert launch['dense_amendment_sha256'] == sha(ROOT/'PREREGISTRATION_C14_DENSE.md')
    for path, digest in launch['model_dependencies'].items():
        assert sha(ROOT.parents[2]/path) == digest
    source = Path(span['source'])
    stat = source.stat()
    assert (stat.st_size,stat.st_ino,stat.st_mtime_ns) == (span['source_size'],span['source_inode'],span['source_mtime_ns'])
    with source.open('rb') as f:
        for sample in span['samples']:
            f.seek(sample['byte_offset'])
            assert hashlib.sha256(f.read(sample['byte_length'])).hexdigest() == sample['sha256']
        for sample in span['span_end_hashes'].values():
            f.seek(sample['offset'])
            assert hashlib.sha256(f.read(sample['bytes'])).hexdigest() == sample['sha256']
    offsets = Path(span['offsets_path'])
    assert sha(offsets) == span['offsets_sha256'] == construction['offsets_sha256']
    index = ROOT.parents[2]/span['documents_index']
    assert sha(index) == span['documents_index_sha256']
    valid = []
    count = total_bytes = 0
    previous_end = 0
    with offsets.open() as f, index.open() as receipts, source.open('rb') as text:
        for line in receipts:
            expected = json.loads(line)
            if expected['truncated']:
                continue
            d = json.loads(next(f))
            assert all(d[k] == v for k,v in expected.items())
            assert not d['truncated'] and d['byte_offset'] >= previous_end
            text.seek(d['byte_offset'])
            assert hashlib.sha256(text.read(d['byte_length'])).hexdigest() == d['sha256']
            previous_end = d['byte_offset']+d['byte_length']
            assert previous_end <= span['span_end_exclusive']
            count += 1
            total_bytes += d['byte_length']
            if d['split'] == 'valid':
                valid.append(d['source_index'])
        assert next(f,None) is None
    assert count == span['matched_documents']-span['excluded_token_truncated']
    halves = evaluation_halves(valid)
    windows = Path(construction['windows_path'])
    assert sha(windows) == construction['windows_sha256']
    expected_counts = {}
    window_count = 0
    with offsets.open() as f, windows.open() as frozen:
        for line in f:
            d = json.loads(line)
            if d['split'] != 'valid':
                continue
            for start in range(0,d['byte_length']-1,128):
                n = min(128,d['byte_length']-1-start)
                row = json.loads(next(frozen))
                assert row == dict(document=d['source_index'],start=start,targets=n,half=halves[d['source_index']])
                key = row['half']
                expected_counts[key] = expected_counts.get(key,0)+n
                window_count += 1
        assert next(frozen,None) is None
    summaries = []
    assert [m['arm'] for m in training['models']] == ['emender','transformer']
    for model, parameters in zip(training['models'],(2133700,2133748)):
        arm = model['arm']
        assert model['parameters'] == parameters and model['cross_document_windows'] == 0
        assert 2700 <= model['active_seconds'] < 2760 and model['steps'] > 0
        assert all(math.isfinite(p['nll']) and p['nll'] >= 0 for p in model['curve'])
        assert sha(windows.parent/f'{arm}_dense.pt') == model['checkpoint_sha256']
        predictions = windows.parent/f'{arm}_eval.jsonl'
        assert sha(predictions) == model['eval_sha256']
        sums, counts, strata = {}, {}, {}
        with windows.open() as expected, predictions.open() as actual:
            for line in expected:
                w = json.loads(line)
                row = json.loads(next(actual))
                assert (row['document'],row['start'],len(row['nll'])) == (w['document'],w['start'],w['targets'])
                losses = row['nll']
                assert all(math.isfinite(x) and x >= 0 for x in losses)
                half = w['half']
                sums[half] = sums.get(half,0.)+sum(losses)
                counts[half] = counts.get(half,0)+len(losses)
                for label,part in [('window_reset_1_16',losses[:16]),('window_interior_17_128',losses[16:])]:
                    key = half+'/'+label
                    s,n = strata.get(key,(0.,0))
                    strata[key] = s+sum(part),n+len(part)
            assert next(actual,None) is None
        assert counts == expected_counts
        replay_errors=[]
        for half,n in counts.items():
            assert model['heldout'][half]['targets'] == n
            error=abs(sums[half]/n-model['heldout'][half]['nll'])
            replay_errors.append(error)
            assert error < 1e-8
        for key,(s,n) in strata.items():
            reported = model['within_document_strata'][key]
            assert reported['targets'] == n
            if n:
                error=abs(s/n-reported['nll'])
                replay_errors.append(error)
                assert error < 1e-8
            else:
                assert reported['nll'] is None
        summaries.append(dict(arm=arm,windows=window_count,targets=counts,max_nll_replay_error=max(replay_errors),nll={k:sums[k]/n for k,n in counts.items()}))
        print(arm,'loss replay passed',flush=True)
    assert json.loads((OUT/'terminal.json').read_text())['exit_code'] == 0
    return dict(status='passed',documents=count,selected_bytes=total_bytes,
                eval_windows=window_count,models=summaries,source_materialized_bytes=0,
                notes='Independent aggregation of every recorded target loss and stratum (not model inference replay); every admitted document digest and span end hashes rechecked. Consultation predictions remain untested.')


if __name__ == '__main__':
    try:
        result = validate()
    except Exception as error:
        (OUT/'validation.json').write_text(json.dumps(dict(status='failed',error=repr(error)),indent=2)+'\n')
        raise
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
