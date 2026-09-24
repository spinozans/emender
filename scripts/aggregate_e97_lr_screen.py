#!/usr/bin/env python3
"""Require complete, identity-consistent LR screening shards before aggregation."""
import argparse
import hashlib
import json
from pathlib import Path
from ndm.e97_atomic import publish_bytes_no_replace


def aggregate(root, panel, panel_sha256):
    shards = [json.loads((root/f'rank-{i:02d}.json').read_text()) for i in range(8)]
    first = shards[0]
    keys = ('checkpoint_sha256', 'weight_mode', 'panel_sha256', 'world_size',
            'controller_closure_sha256', 'evaluator_sha256')
    for rank, shard in enumerate(shards):
        assert shard['rank']==rank and shard['world_size']==8
        assert all(shard[k]==first[k] for k in keys)
        assert shard['panel_sha256']==panel_sha256
        assert [r['id'] for r in shard['results']]==[t['id'] for t in panel['tasks'][rank::8]]
    results = [r for shard in shards for r in shard['results']]
    assert len(results)==24 and len({r['id'] for r in results})==24
    return {'schema': 'emender-e97-lr-screen-summary-v1',
            **{k: first[k] for k in keys}, 'tasks': len(results),
            'passed': sum(r['success'] for r in results),
            'first_turn_protocol_valid': sum(r['first_turn_protocol_valid'] for r in results),
            'tool_tasks_passed': sum(r['success'] for r in results if r['kind'] in {'read','two_reads','recovery'}),
            'completion_tokens': sum(t['completion_tokens'] for r in results for t in r['turns']),
            'tool_calls': sum(len(r['reads'])+len(r['missing_reads']) for r in results),
            'by_kind': {kind: {'tasks': sum(r['kind']==kind for r in results),
                               'passed': sum(r['success'] for r in results if r['kind']==kind)}
                        for kind in sorted({r['kind'] for r in results})},
            'results': results, 'training_eligible': False,
            'scope': 'fresh-instance direct-engine development screening, not final holdout or HTTP/Pi qualification'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    payload=args.panel.read_bytes()
    result=aggregate(args.input_root, json.loads(payload), hashlib.sha256(payload).hexdigest())
    publish_bytes_no_replace(args.output, (json.dumps(result,indent=2,sort_keys=True)+'\n').encode(), mode=0o600)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}))

if __name__=='__main__':
    main()
