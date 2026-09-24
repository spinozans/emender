#!/usr/bin/env python3
"""Aggregate supplied-opening diagnostics without autonomous-success labels."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from ndm.e97_atomic import publish_bytes_no_replace


def aggregate(root,panel_path,world=8):
    payload=panel_path.read_bytes(); panel=json.loads(payload)
    paths=sorted(root.glob('rank-*.json'))
    if len(paths)!=world: raise ValueError('incomplete shard set')
    reports=[json.loads(p.read_text()) for p in paths]
    keys=('schema','intervention','checkpoint_sha256','weight_mode','panel_sha256','world_size','controller_closure_sha256','evaluator_sha256')
    identity={k:reports[0][k] for k in keys}
    if identity['schema']!='emender-e97-analysis-prefill-shard-v1' or identity['world_size']!=world or identity['panel_sha256']!=hashlib.sha256(payload).hexdigest():
        raise ValueError('invalid prefill identity')
    results=[]
    for rank,report in enumerate(reports):
        if {k:report[k] for k in keys}!=identity or report['rank']!=rank or report['device']!=f'cuda:{rank}' or report['training_eligible'] is not False:
            raise ValueError('mixed identities or invalid device routing')
        if [r['id'] for r in report['results']]!=[t['id'] for t in panel['tasks'][rank::world]]:
            raise ValueError('invalid shard membership')
        for r in report['results']:
            if 'first_turn_protocol_valid' in r: raise ValueError('autonomous protocol field must not be reused')
            for turn in r['turns']:
                intervention=turn['prefill_intervention']
                if (intervention['prefill']!='Analysis:' or intervention['prefill_token_ids']!=[32750,25]
                        or not 1<=intervention['model_generated_tokens']<=4094
                        or turn['completion_tokens']!=len(intervention['prefill_token_ids'])+intervention['model_generated_tokens']):
                    raise ValueError('invalid prefill accounting')
        results.extend(report['results'])
    expected={t['id'] for t in panel['tasks']}
    if len(results)!=len(expected) or {r['id'] for r in results}!=expected: raise ValueError('duplicate/missing results')
    kinds=defaultdict(lambda:{'tasks':0,'prefilled_successes':0})
    for r in results:
        kinds[r['kind']]['tasks']+=1; kinds[r['kind']]['prefilled_successes']+=int(r['success'])
    turns=[t for r in results for t in r['turns']]
    return {'schema':'emender-e97-analysis-prefill-summary-v1','training_eligible':False,'identity':identity,
            'tasks':len(results),'prefilled_successes':sum(r['success'] for r in results),
            'prefilled_first_turn_protocol_valid':sum(r['prefilled_first_turn_protocol_valid'] for r in results),
            'tool_tasks_passed':sum(r['success'] and r['kind'] in {'read','two_reads','recovery'} for r in results),
            'admitted_read_calls':sum(len(r['reads'])+len(r['missing_reads']) for r in results),
            'model_generated_tokens':sum(t['prefill_intervention']['model_generated_tokens'] for t in turns),
            'supplied_tokens':sum(len(t['prefill_intervention']['prefill_token_ids']) for t in turns),
            'stop_reasons':dict(Counter(t['prefill_intervention']['stop_reason'] for t in turns)),
            'by_kind':dict(kinds),'results':results,
            'shards':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
            'scope':'diagnostic with externally supplied opening on consumed development instances; not autonomous capability or promotion'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); result=aggregate(args.input_root,args.panel)
    publish_bytes_no_replace(args.output,(json.dumps(result,indent=2,sort_keys=True)+'\n').encode(),mode=0o600)
    print(json.dumps({k:v for k,v in result.items() if k not in {'results','shards'}}),flush=True)

if __name__=='__main__': main()
