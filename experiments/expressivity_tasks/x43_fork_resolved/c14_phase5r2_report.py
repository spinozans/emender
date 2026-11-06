"""Recompute round2 measurements and explicitly limited scenario projection."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',default='stream_v1');a=p.parse_args()
    r=Path(__file__).resolve().parent/'receipts/scalability_phase5_round2'
    data=json.loads((r/f'{a.label}.json').read_text())
    assert len(data['rows'])==10
    for name,expected in data['sources'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    assert all(row['exact'] and row['repeat_exact'] for row in data['rows'])
    row=data['rows'][-1];ms=row['median_ms']
    projection=dict(native_head_ms=ms,matched_reference_ms=row['reference_median_ms'],
        matched_speedup=row['reference_median_ms']/ms,compiler_spills=row['spills'],
        seconds_per_target=ms/1000*22*2*2/1024,
        native_only_days=ms/1000*22*2*2/1024*2e9/86400,
        required_targets_per_second=2e9/86400,
        max_head_ms_if_other_work_free=86400/2e9*1024/(22*2*2)*1000,
        status='NO-GO for 1.3B/24h; internal long-context speed improvement only',
        scope='Hypothetical 22-layer/two-head/two-pass, 2B raw-byte budget scenario; excludes replay/dense/optimizer/data; no sustained training measurement')
    projection['targets_per_second']=1/projection['seconds_per_target']
    (r/f'{a.label}_projection.json').write_text(json.dumps(projection,indent=2)+'\n')
    print(json.dumps(projection,indent=2))
    print('\n'.join(f"|{x['batch']}|{x['tokens']}|{x['reference_median_ms']:.6f}|{x['median_ms']:.6f}|{x['registers']}|{x['spills']}|" for x in data['rows']))

if __name__=='__main__':main()
